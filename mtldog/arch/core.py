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
    
    def get_share_module(self) -> nn.Module:
        raise NotImplementedError()
    
    def get_heads_module(self) -> nn.ModuleDict:
        raise NotImplementedError()
    
    def get_share_params(self) -> Tensor:
        return self.get_share_module().parameters()
    
    def get_heads_params(self) -> Dict[str, Tensor]:
        return {tk : self.get_heads_module()[tk].parameters() for tk in self.args.tkss}
    
    def get_share_params_require_grad(self) -> List[Tensor]:
        return [p for p in self.get_share_params() if p.grad is not None]

    def get_heads_params_require_grad(self) -> Dict[str, List[Tensor]]:
        return {tk : [p for p in self.get_heads_params()[tk] if p.grad is not None] for tk in self.args.tkss}

    def zero_grad_share_params(self) -> None:
        self.get_share_module().zero_grad()

    def zero_grad_heads_params(self) -> None:
        self.get_heads_module().zero_grad()
    
    def name_share_params_require_grad(self) -> None:
        return [self.name_norm(n, p) for n, p in self.get_share_module().named_parameters() if p.grad is not None]

    def name_heads_params_require_grad(self) -> None:
        return {tk : [self.name_norm(n, p) for n, p in self.get_heads_module()[tk].named_parameters() if p.grad is not None] for tk in self.tkss}

    @staticmethod
    def name_norm(n: str , p: Tensor):
        if len(p.size()) == 4:
            return 'conv.' + n
        elif len(p.size()) == 2:
            return 'lin.' + n
        elif len(p.size()) == 1:
            if 'weight' in n:
                return 'norm.' + n
            else:
                return 'unk.' + n