from torch import Tensor
import torch
import torch.nn.functional as F

def metric_rec_psnr(preds: Tensor, label: Tensor) -> float:
    mse = F.mse_loss(preds, label)

    return (
        20 * torch.log10(
            torch.clamp(
                torch.max(preds) / torch.sqrt(mse), min=0.001
            )
        )
    ).item()