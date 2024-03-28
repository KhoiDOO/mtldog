from torch import Tensor
from argparse import Namespace

import torch.nn.functional as F

def loss_seg_ce(logits: Tensor, labels: Tensor, args: Namespace) -> Tensor:
    return F.cross_entropy(logits, labels)