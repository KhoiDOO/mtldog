from torch import Tensor
from torchmetrics import JaccardIndex

def metric_seg_miou(preds: Tensor, label: Tensor) -> float:
    B, C, H, W = preds.shape

    preds = preds.permute(0, 2, 3, 1).flatten(0, 2).argmax(1)
    label = label.permute(0, 2, 3, 1).flatten(0, 2).argmax(1)

    jaccard = JaccardIndex(task="multiclass", num_classes=C).to(preds.device)

    miou: Tensor = jaccard(preds, label)

    return miou.item()