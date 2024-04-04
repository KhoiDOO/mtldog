import torch.nn.functional as F
from torch import Tensor    
import torch

def metric_cls_acc(pred: Tensor, target: Tensor)->float:
    preds = torch.argmax(pred, dim=1)
    acc = (preds == target).float().mean()
    
    return acc.item()