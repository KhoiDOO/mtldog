from torch import nn, Tensor
from argparse import Namespace
from .mnist import GlobalAvgPooling

import torch
import torch.nn.functional as F

class MNIST_HPS_AE_DEBUG_ENCODER(nn.Module):
    def __init__(self, args:Namespace) -> nn.Module:
        super(MNIST_HPS_AE_DEBUG_ENCODER, self).__init__()

        self.net = torch.nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 4),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 8),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 8),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 8)
        )

    def forward(self, x: Tensor):
        return self.net(x)

class MNIST_HPS_CLS_AE_DEBUG_DECODER(nn.Module):
    def __init__(self, args:Namespace) -> nn.Module:
        super(MNIST_HPS_CLS_AE_DEBUG_DECODER, self).__init__()

        self.args = args

        self.avgpool = GlobalAvgPooling()
        self.classifier = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, args.num_class)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        lat = self.avgpool(x)
        return self.classifier(lat)

class MNIST_HPS_REC_AE_DEBUG_DECODER(nn.Module):
    def __init__(self, args:Namespace) -> nn.Module:
        super(MNIST_HPS_REC_AE_DEBUG_DECODER, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 8),
            nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 8),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.GroupNorm(2, 4),
            nn.ConvTranspose2d(4, 1, 3, stride=1, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def arch_mnist_hps_ae_debug_encoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_AE_DEBUG_ENCODER(args)

def arch_mnist_hps_cls_ae_debug_decoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_CLS_AE_DEBUG_DECODER(args)

def arch_mnist_hps_rec_ae_debug_decoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_REC_AE_DEBUG_DECODER(args)

def arch_mnistmed_hps_ae_debug_encoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_AE_DEBUG_ENCODER(args)

def arch_mnistmed_hps_cls_ae_debug_decoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_CLS_AE_DEBUG_DECODER(args)

def arch_mnistmed_hps_rec_ae_debug_decoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_REC_AE_DEBUG_DECODER(args)

def arch_mnisteasy_hps_ae_debug_encoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_AE_DEBUG_ENCODER(args)

def arch_mnisteasy_hps_cls_ae_debug_decoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_CLS_AE_DEBUG_DECODER(args)

def arch_mnisteasy_hps_rec_ae_debug_decoder(args: Namespace) -> nn.Module:
    return MNIST_HPS_REC_AE_DEBUG_DECODER(args)