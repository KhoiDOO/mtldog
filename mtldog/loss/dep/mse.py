from torch import Tensor
from argparse import Namespace

import torch.nn.functional as F

def loss_depth_mse(logits: Tensor, labels: Tensor, args: Namespace) -> Tensor:
    return F.mse_loss(logits, labels)