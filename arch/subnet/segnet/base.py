from .core import * 
from typing import Tuple
from torch import nn, Tensor
from argparse import Namespace

class SegNetEncoder(nn.Module):
    def __init__(self, init_ch: int) -> nn.Module:
        super(SegNetEncoder, self).__init__()

        self.init_ch = init_ch

        self.downconv0 = DownConv2(3, self.init_ch, kernel_size=3)
        self.downconv1 = DownConv2(self.init_ch, self.init_ch*2, kernel_size=3)
        self.downconv2 = DownConv3(self.init_ch*2, self.init_ch*4, kernel_size=3)
        self.downconv3 = DownConv3(self.init_ch*4, self.init_ch*8, kernel_size=3)
    
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x, mp0_indices, shape0 = self.downconv0(x)
        x, mp1_indices, shape1 = self.downconv1(x)
        x, mp2_indices, shape2 = self.downconv2(x)
        x, mp3_indices, shape3 = self.downconv3(x)

        return x, (mp0_indices, mp1_indices, mp2_indices, mp3_indices), (shape0, shape1, shape2, shape3)

class SegNetDecoder(nn.Module):
    def __init__(self, init_ch, seg_num_classes) -> None:
        super(SegNetDecoder, self).__init__()

        self.seg_num_class = seg_num_classes
        self.init_ch = init_ch

        self.upconv0 = UpConv3(self.init_ch*8, self.init_ch*4, kernel_size=3)
        self.upconv1 = UpConv3(self.init_ch*4, self.init_ch*2, kernel_size=3)
        self.upconv2 = UpConv2(self.init_ch*2, self.init_ch, kernel_size=3)
        self.upconv3 = UpConv2(self.init_ch, self.seg_num_class, kernel_size=3)
    
    def forward(self, input: Tensor) -> Tuple[Tensor]:
        x, (mp0_indices, mp1_indices, mp2_indices, mp3_indices), (shape0, shape1, shape2, shape3) = input

        x = self.upconv0(x, mp3_indices, output_size=shape3)
        x = self.upconv1(x, mp2_indices, output_size=shape2)
        x = self.upconv2(x, mp1_indices, output_size=shape1)
        masks = self.upconv3(x, mp0_indices, output_size=shape0)

        return masks