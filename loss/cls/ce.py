from torch import Tensor
from argparse import Namespace

import torch.nn.functional as F

def loss_cls_ce(logits: Tensor, labels: Tensor, args: Namespace) -> Tensor:
    return F.cross_entropy(logits, labels)