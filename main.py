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

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss, CrossBatchMemory

from xbm import XBM
from datasets import get_dataset
from engine import train, evaluate
from regularizer import DifferentialEntropyRegularization


def get_args_parser():
    parser = argparse.ArgumentParser('Training Vision Transformers for Image Retrieval', add_help=False)

    parser.add_argument('--max-iter', default=2_000, type=int)
    parser.add_argument('--batch-size', default=64, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')

    # Learning rate schedule parameters
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (3e-5 for category level)')

    # Dataset parameters
    parser.add_argument('--data-set', default='sop', choices=['cub200', 'sop', 'inshop'],
                        type=str, help='dataset path')
    parser.add_argument('--data-path', default='./data/Stanford_Online_Products', type=str,
                        help='dataset path')
    parser.add_argument('--rank', default=[1, 2, 4, 8], nargs="+", type=int, help="compute recall@r for each r")

    # Loss parameters
    parser.add_argument('--lambda-reg', type=float, default=0.7,
                        help="strength of differential entropy regularization(lambda)")
    parser.add_argument('--margin', type=float, default=0.5,
                        help="negative margin of contrastive loss(beta)")

    # XBM parameters
    parser.add_argument('--xbm-warmup-steps', type=int, default=0, help="activate xbm after 1,000 steps")
    parser.add_argument('--memory-ratio', type=float, default=1.0, help="size of the xbm queue")
    parser.add_argument('--xbm-random-init', action='store_true',
                        help="if set true, initialize queue with randomly sampled training images after warmup")
    # parser.set_defaults(xbm_random_init=True)
    parser.add_argument('--encoder-momentum', type=float, default=None,
                        help="momentum for the key encoder (0.999 for In-Shop dataset)")

    # Dataset parameters
    parser.add_argument('--output-dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--log-dir', default='./logs', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')

    parser.set_defaults(pin_mem=True)

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
    logging.info(f"number of training examples: {len(dataset_train)}")
    logging.info(f"number of query examples: {len(dataset_query)}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
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
    logging.info(f"Creating model: {args.model}")
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
    logging.info(f'number of params: {round(n_parameters / 1_000_000, 2):.2f} M')

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
    logging.info(f"criterion: {criterion}")

    log_writer = None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    logging.info(f"Start training for {args.max_iter:,} iterations")
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
        args.log_dir = os.path.join(args.log_dir, args.data_set)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.data_set)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
