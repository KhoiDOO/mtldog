from torch import nn, Tensor
from argparse import Namespace

import torch
import torch.nn.functional as F

class MNIST_HPS_AE_BASE_ENCODER(nn.Module):
    def __init__(self, args:Namespace) -> nn.Module:
        super(MNIST_HPS_AE_BASE_ENCODER, self).__init__()

        self.net = torch.nn.Sequential(
            nn.Conv2d(1, 64, 3, strid=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128)
        )

    def forward(self, x: Tensor):
        return self.net(x)

class MNIST_HPS_CLS_AE_BASE_DECODER(nn.Module):
    def __init__(self, args:Namespace) -> nn.Module:
        super(MNIST_HPS_CLS_AE_BASE_DECODER, self).__init__()

        self.args = args

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, args.num_class)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        lat = self.avgpool(x)
        return self.classifier(lat)

class MNIST_HPS_REC_AE_BASE_DECODER(nn.Module):
    def __init__(self, args:Namespace) -> nn.Module:
        super(MNIST_HPS_REC_AE_BASE_DECODER, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)