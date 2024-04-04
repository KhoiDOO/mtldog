from argparse import Namespace
from torch import Tensor
from .core import MTLDOGALGO
from typing import Dict

import torch


class AVG_GRAD(MTLDOGALGO):
    def __init__(self):
        super().__init__()

    def backward(self, losses: Dict[str, Tensor]):
        
        share_grads = torch.stack([self.get_grads_share(loss, mode='backward') for loss in losses.values()])
        mean_share_grads = torch.mean(share_grads, dim=[0,1])
        # mean_share_grads = share_grads[0][0]
        
        head_grad = []
        for loss in losses.values():
            head_grad.append(self.get_grads_heads(loss, mode='backward'))
        mean_head_grad = {head: torch.mean(torch.stack([grad[head] for grad in head_grad]), dim=0).to(device=self.device) for head in self.args.tkss}
        
        self.reset_grad_share(mean_share_grads)
        self.reset_grad_heads(mean_head_grad)
        
        return None, None

def algo_avg_grad():
    return AVG_GRAD