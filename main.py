import argparse
import datetime
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from timm.models import create_model
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import RandomSampler
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss

from xbm import XBM
from datasets import get_dataset
from engine import train, evaluate
from regularizer import DifferentialEntropyRegularization


def get_args_parser():
    parser = argparse.ArgumentParser('Training Vision Transformers for Image Retrieval', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='deit_small_distilled_patch16_224', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--max-iter', default=2_000, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (3e-5 for category level)')
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')

    # Dataset parameters
    parser.add_argument('--dataset', default='cub200', choices=['cub200', 'sop', 'inshop'], type=str, help='dataset path')
    parser.add_argument('--data-path', default='/data/CUB_200_2011', type=str, help='dataset path')
    parser.add_argument('--m', default=0, type=int, help="sample m images per class")
    parser.add_argument('--rank', default=[1, 2, 4, 8], nargs="+", type=int, help="compute recall@r")
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Loss parameters
    parser.add_argument('--lambda-reg', type=float, default=0.7, help="regularization strength")
    parser.add_argument('--margin', type=float, default=0.5,
                        help="negative margin of contrastive loss(beta)")

    # xbm parameters
    parser.add_argument('--memory-ratio', type=float, default=1.0, help="size of the xbm queue")
    parser.add_argument('--encoder-momentum', type=float, default=None,
                        help="momentum for the key encoder (0.999 for In-Shop dataset)")

    # MISC
    parser.add_argument('--logging-freq', type=int, default=50)
    parser.add_argument('--output-dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--log-dir', default='./logs', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    return parser


def main(args):

    logging.info("=" * 20 + " training arguments " + "=" * 20)
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
    logging.info("=" * 60)

    # fix random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # get training/query/gallery dataset
    dataset_train, dataset_query, dataset_gallery = get_dataset(args)
    logging.info(f"Number of training examples: {len(dataset_train)}")
    logging.info(f"Number of query examples: {len(dataset_query)}")

    sampler_train = RandomSampler(dataset_train)
    if args.m: sampler_train = MPerClassSampler(dataset_train.labels, m=args.m, batch_size=args.batch_size)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )

    data_loader_gallery = None
    if dataset_gallery is not None:
        data_loader_gallery = torch.utils.data.DataLoader(
            dataset_gallery,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )

    # get model
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=0,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    momentum_encoder = None
    if args.encoder_momentum is not None:
        momentum_encoder = create_model(
            args.model,
            num_classes=0,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
        for param_q, param_k in zip(model.parameters(), momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        momentum_encoder.to(device)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of params: {round(n_parameters / 1_000_000, 2):.2f} M')

    # get optimizer
    optimizer = create_optimizer(args, model)

    # get loss & regularizer
    criterion = ContrastiveLoss(
        pos_margin=1.0,
        neg_margin=args.margin,
        distance=CosineSimilarity(),
    )
    regularization = DifferentialEntropyRegularization()
    xbm = XBM(
        memory_size=int(len(dataset_train) * args.memory_ratio),
        embedding_dim=model.embed_dim,
        device=device
    )
    loss_scaler = NativeScaler()

    log_writer = None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    start_time = time.time()
    train(
        model,
        momentum_encoder,
        criterion,
        xbm,
        regularization,
        data_loader_train,
        optimizer,
        device,
        loss_scaler,
        args.clip_grad,
        log_writer,
        args=args
    )

    logging.info("Start evaluation job")

    evaluate(
        data_loader_query,
        data_loader_gallery,
        model,
        device,
        rank=sorted(args.rank)
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser('Training Vision Transformers for Image Retrieval', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, args.dataset)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.dataset)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
