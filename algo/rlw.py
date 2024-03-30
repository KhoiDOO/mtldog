from argparse import Namespace
from torch import Tensor
from .core import MTLDOGALGO
from typing import Dict

import torch
import torch.nn.functional as F


class RLW(MTLDOGALGO):
    def __init__(self):
        super().__init__()

    def backward(self, losses: Dict[str, Tensor]):
        task_loss = torch.mean(torch.cat(list(losses.values())), dim = 0)

        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)

        total_loss = torch.mul(task_loss, batch_weight).sum()

        total_loss.backward()

        return None, None

def algo_rlw():
    return RLW