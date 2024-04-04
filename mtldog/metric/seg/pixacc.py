from torch import Tensor

def metric_seg_pixacc(pred_mask: Tensor, label_mask: Tensor) -> float:
    B, _, H, W = label_mask.shape
    return (pred_mask.argmax(dim=1) == label_mask.argmax(dim=1)).sum().item() / (B * H * W)