from argparse import Namespace
from torch import Tensor
from .core import MTLDOGALGO
from typing import Dict

import torch


class ERM(MTLDOGALGO):
    def __init__(self):
        super().__init__()

    def backward(self, losses: Dict[str, Tensor]):
        total_loss = torch.sum(sum(losses.values()))
        total_loss.backward()

        return None, None

def algo_erm():
    return ERM