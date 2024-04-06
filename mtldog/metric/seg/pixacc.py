from torch import Tensor
from torchmetrics import Accuracy
import torch

def metric_seg_pixacc(preds: Tensor, label: Tensor) -> float:
    B, C, H, W = preds.shape

    preds = preds.permute(0, 2, 3, 1).flatten(0, 2).argmax(1)
    label = label.permute(0, 2, 3, 1).flatten(0, 2).argmax(1)

    accuracy = Accuracy(task="multiclass", num_classes=C).to(preds.device)

    acc: Tensor = accuracy(preds, label)

    return acc.item()