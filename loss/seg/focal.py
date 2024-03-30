from torch import Tensor
from argparse import Namespace

import torch.nn.functional as F
import torch

def loss_seg_focal(logits: Tensor, labels:Tensor, args: Namespace) -> Tensor:
    
    log_prob = F.log_softmax(logits, dim=1)

    entropy = torch.pow(1 - log_prob.exp(), args.gamma) * log_prob * labels

    loss = (-1) * entropy.mean()

    return loss

