from torch import Tensor
import torch

def depth_rel_error(x_pred: Tensor, x_output: Tensor) -> float:
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)

    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true

    return (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()