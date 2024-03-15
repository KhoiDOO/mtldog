import torch.nn.functional as F
import torch
from torch import Tensor
from argparse import Namespace

def loss_cls_focal(logits: Tensor, labels:Tensor, args: Namespace):
    
    prob = F.log_softmax(logits, dim=1)

    _, C = tuple(logits.size())

    entropy = torch.pow(1 - prob.exp(), args.gamma) * logits * F.one_hot(labels, num_classes=C).float()

    loss = (-1) * entropy.mean()

    return loss