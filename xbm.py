# XBM implementation from : https://github.com/msight-tech/research-xbm
# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
# ---------
# And momentum encoder implementation from : https://github.com/facebookresearch/moco

import torch
import torch.nn.functional as F


class XBM:
    def __init__(self, memory_size, embedding_dim, device):
        self.K = memory_size
        self.feats = torch.zeros(self.K, embedding_dim).to(device)
        self.targets = -torch.ones(self.K, dtype=torch.long).to(device)
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size


@torch.no_grad()
def initialize_xbm(
        xbm: XBM,
        encoder: torch.nn.Module,
        data_loader:torch.utils.data.DataLoader,
        device: torch.device
):
    """Initialize XBM queue with randomly selected images"""
    for images, targets in data_loader:
        if xbm.is_full:
            break
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                features = encoder.forward_features(images)
                if isinstance(features, tuple):
                    features = features[0]
                features = F.normalize(features, dim=1)
            xbm.enqueue_dequeue(features, targets)


@torch.no_grad()
def momentum_update_key_encoder(encoder_q, encoder_k, m=0.999):
    """Momentum update of the key encoder"""
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)
