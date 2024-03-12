import torch
from torch import nn


class Core(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.task_num = args.task_num

        self.encoder = None
        self.decoder = None
    
    def forward(self, x):
        raise NotImplementedError()
    
    def get_share_params(self):
        raise NotImplementedError()

    def zero_grad_share_params(self):
        raise NotImplementedError()