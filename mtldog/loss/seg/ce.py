from torch import Tensor
from argparse import Namespace

import torch
import torch.nn.functional as F

def loss_seg_ce(logits: Tensor, labels: Tensor, args: Namespace) -> Tensor:
    return F.cross_entropy(logits, labels)

def loss_seg_gumce(logits: Tensor, labels: Tensor, args: Namespace) -> Tensor:
    log_prob = F.log_softmax(logits / args.tau)

    B, _, H, W = tuple(logits.size())

    entropy = log_prob * labels

    return (-1 / (B * H * W)) * torch.sum(entropy)