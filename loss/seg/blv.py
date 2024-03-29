from torch import Tensor
from argparse import Namespace
from torch.distributions import normal

import torch.nn.functional as F
import torch

def loss_seg_blv(logits: Tensor, labels:Tensor, args: Namespace) -> Tensor:
    
    m_list = torch.sum(labels, dim=[0, 2, 3]) + 0.0000001

    frequency_list = torch.log(m_list)

    viariation = normal.Normal(0, args.blv_s).sample(logits.shape).clamp(-1, 1).to(logits.device)

    logits = logits + (viariation.abs().permute(0, 2, 3, 1) / frequency_list.max() * frequency_list).permute(0, 3, 1, 2)

    loss = F.cross_entropy(logits, labels)

    return loss