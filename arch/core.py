from torch import Tensor
from torch import nn
from typing import List, Dict

class MTLDOGARCH(nn.Module):
    def __init__(self, args) -> nn.Module:
        super().__init__()
        self.args = args
        self.task_num = args.task_num
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def get_share_params(self) -> Tensor[Tensor]:
        raise NotImplementedError()
    
    def get_heads_params(self) -> Dict[str, Tensor[Tensor]]:
        raise NotImplementedError()

    def zero_grad_share_params(self) -> None:
        raise NotImplementedError()

    def zero_grad_heads_params(self) -> None:
        raise NotImplementedError()