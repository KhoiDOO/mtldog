from torch import Tensor
import torch
import torch.nn.functional as F

def metric_rec_psnr(pred: Tensor, target: Tensor) -> float:
    mse = F.mse_loss(pred, target)

    return (20 * torch.log10(torch.max(pred) / torch.sqrt(mse))).item()