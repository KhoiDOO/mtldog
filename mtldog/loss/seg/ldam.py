from torch import Tensor
from argparse import Namespace

import torch.nn.functional as F
import torch

def loss_seg_ldam(logits: Tensor, labels:Tensor, args: Namespace) -> Tensor:

    m_list = torch.sum(labels, dim=[0, 2, 3]) + 0.0000001
    m_list = 1.0 / torch.sqrt(torch.sqrt(m_list))
    m_list = m_list * (args.ldamm / torch.max(m_list))

    _logits = logits.permute(0, 2, 3, 1).flatten(0, -2)
    _labels = labels.permute(0, 2, 3, 1).flatten(0, -2).argmax(1).long()

    index = torch.zeros_like(_logits, dtype=torch.uint8)
    index.scatter_(1, _labels.data.view(-1, 1), 1)
    
    index_float = index.type(torch.cuda.FloatTensor)
    batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
    batch_m = batch_m.view((-1, 1))
    x_m = _logits - batch_m

    output = torch.where(index, x_m, _logits)
    return F.cross_entropy(args.ldams * output, _labels)