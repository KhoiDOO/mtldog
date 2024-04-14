from argparse import Namespace
from torch import nn, Tensor
from .core import MTLDOGALGO
from typing import Dict

import torch


class UW(MTLDOGALGO):
    def __init__(self):
        super().__init__()

    def init_param(self, args: Namespace) -> None:
        super().init_param(args)

        self.loss_scale = nn.Parameter(torch.tensor([-0.5]*self.task_num, device=self.device))

    def backward(self, losses: Dict[str, Tensor]):
        task_loss = torch.mean(torch.cat(list(losses.values())), dim = 0)
        total_loss = (task_loss/(2*self.loss_scale.exp())+self.loss_scale/2).sum()
        total_loss.backward(retain_graph=True)

        return None, None

    def get_grads_dm_share_heads(self, losses: Dict[str, Tensor], detach: bool) -> Dict[str, Tensor | Dict[str, Tensor]]:
        grad_dict = super().get_grads_dm_share_heads(losses, detach)

        self.loss_scale.grad = None

        return grad_dict

def algo_uw():
    return UW