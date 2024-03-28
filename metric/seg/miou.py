from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np

def metric_seg_miou(pred_label_mask: Tensor, label_mask: Tensor) -> float:
    
    smooth=1e-10
    _, n_classes, _, _ = pred_label_mask.shape
    
    pred_label_mask = F.softmax(pred_label_mask, dim=1)
    pred_label_mask = torch.argmax(pred_label_mask, dim=1)
    pred_label_mask = pred_label_mask.contiguous().view(-1)
    label_mask = torch.argmax(label_mask, dim=1)
    label_mask = label_mask.contiguous().view(-1)

    iou_per_class = []
    for clas in range(0, n_classes): #loop per pixel class
        true_class = pred_label_mask == clas
        true_label = label_mask == clas

        if true_label.long().sum().item() == 0: #no exist label in this loop
            iou_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()

            iou = (intersect + smooth) / (union + smooth)
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class).item()