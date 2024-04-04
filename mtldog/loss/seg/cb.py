from torch import Tensor
from argparse import Namespace

import torch.nn.functional as F
import torch

def loss_seg_cb(logits: Tensor, labels:Tensor, args: Namespace) -> Tensor:
    
    log_prob = F.log_softmax(logits, dim=1)

    cls_loss = {}

    B, C, H, W = tuple(logits.size())

    _log_prob = log_prob.permute(0, 2, 3, 1).flatten(0, -2)
    _labels = labels.permute(0, 2, 3, 1).flatten(0, -2)
    
    N, _ = tuple(_labels.shape)
    
    beta = (N - 1)/N

    for cidx in range(C):
        c_logprob = _log_prob[_labels[:, cidx] == 1]
        c_target = _labels[_labels[:, cidx] == 1]
        
        N_c, _ = tuple(c_target.shape)

        entropy = ((1 - beta)/(1 - beta ** N_c + 1e-6)) * torch.sum(c_logprob * c_target)

        cls_loss[cidx] = entropy

    return (-1 / (B * H * W)) * sum(list(cls_loss.values()))

def loss_seg_cbfocal(logits: Tensor, labels:Tensor, args: Namespace) -> Tensor:

    log_prob = F.log_softmax(logits, dim=1)

    cls_loss = {}

    B, C, H, W = tuple(logits.size())

    _log_prob = log_prob.permute(0, 2, 3, 1).flatten(0, -2)
    _labels = labels.permute(0, 2, 3, 1).flatten(0, -2)
    
    N, _ = tuple(_labels.shape)
    
    beta = (N - 1)/N

    for cidx in range(C):
        c_logprob = _log_prob[_labels[:, cidx] == 1]
        c_target = _labels[_labels[:, cidx] == 1]
        
        N_c, _ = tuple(c_target.shape)

        entropy = ((1 - beta)/(1 - beta ** N_c + 1e-6)) * torch.sum(torch.pow(1 - c_logprob.exp(), args.gamma) * c_logprob * c_target)

        cls_loss[cidx] = entropy

    return (-1 / (B * H * W)) * sum(list(cls_loss.values()))