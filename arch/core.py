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
    
    def get_share_params(self) -> Tensor:
        raise NotImplementedError()
    
    def get_heads_params(self) -> Dict[str, Tensor]:
        raise NotImplementedError()
    
    def get_share_params_require_grad(self) -> List[Tensor]:
        return [p for p in self.get_share_params() if p.grad is not None]

    def get_heads_params_require_grad(self) -> Dict[str, List[Tensor]]:
        return {tk : [p for p in self.get_heads_params()[tk] if p.grad is not None] for tk in self.args.tkss}

    def zero_grad_share_params(self) -> None:
        raise NotImplementedError()

    def zero_grad_heads_params(self) -> None:
        raise NotImplementedError()
    
    def name_share_params_require_grad(self) -> None:
        raise NotImplementedError()

    def name_heads_params_require_grad(self) -> None:
        raise NotImplementedError()