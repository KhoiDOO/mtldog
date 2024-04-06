from .core import * 
from typing import Tuple
from torch import nn, Tensor
from argparse import Namespace

class SegNetEncoder(nn.Module):
    def __init__(self, init_ch: int, depth: int = 4) -> nn.Module:
        super(SegNetEncoder, self).__init__()

        self.init_ch = init_ch
        self.depth = depth

        self.sub_modules = []

        for dep in range(-1, depth - 1):
            lower = int(self.init_ch*(2**dep) if dep != -1 else 3)
            higher = int(self.init_ch*(2**(dep + 1)))
            self.sub_modules.append(DownConv2(lower, higher, kernel_size=3))
        
        self.sub_modules = nn.ModuleList(self.sub_modules)
    
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        mp_indices = []
        shapes = []

        for i, l in enumerate(self.sub_modules):
            x, mp_index, shape = l(x)
            mp_indices.append(mp_index)
            shapes.append(shape)
        
        return x, mp_indices, shapes


class SegNetDecoder(nn.Module):
    def __init__(self, init_ch: int, depth: int, seg_num_classes: int) -> None:
        super(SegNetDecoder, self).__init__()

        self.seg_num_class = seg_num_classes
        self.init_ch = init_ch
        self.depth = depth

        self.sub_modules = []

        for dep in reversed(range(-1, depth - 1)):
            higher = int(self.init_ch*(2**(dep + 1)))
            lower = int(self.seg_num_class if dep == -1 else self.init_ch*(2**dep))
            self.sub_modules.append(UpConv3(higher, lower, kernel_size=3))
        
        self.sub_modules = nn.ModuleList(self.sub_modules)
    
    def forward(self, input: Tensor) -> Tuple[Tensor]:

        x, mp_indices, shapes = input

        for i, (l, m, s) in enumerate(zip(self.sub_modules, reversed(mp_indices), reversed(shapes))):
            x = l(x, m, s)

        return x