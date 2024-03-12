from torch import Tensor
from torch import nn


class MTLDOGARCH(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.task_num = args.task_num
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def get_share_params(self):
        raise NotImplementedError()

    def zero_grad_share_params(self):
        raise NotImplementedError()