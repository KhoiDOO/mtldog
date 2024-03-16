from argparse import Namespace
from torch import Tensor
from .core import MTLDOGALGO

import torch


class ERM(MTLDOGALGO):
    def __init__(self):
        super().__init__()

    def backward(self, losses: Tensor):
        total_loss = torch.sum(losses)
        total_loss.backward()

        return None

def algo_erm():
    return ERM