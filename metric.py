"""
recall@k implementation from : https://github.com/leftthomas/CGD
"""

import torch
from typing import List


def recall(query_features, query_labels, rank: List[int], gallery_features=None, gallery_labels=None):
    num_querys = len(query_labels)
    gallery_features = query_features if gallery_features is None else gallery_features

    cosine_matrix = query_features @ gallery_features.t()

    if gallery_labels is None:
        cosine_matrix.fill_diagonal_(-float('inf'))
        gallery_labels = query_labels

    idx = cosine_matrix.topk(k=rank[-1], dim=-1, largest=True)[1]
    recall_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == query_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        recall_list.append((torch.sum(correct) / num_querys).item())
    return recall_list
