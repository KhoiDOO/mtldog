from argparse import Namespace
from torch import Tensor
from .core import MTLDOGALGO
from typing import Dict
import numpy as np

import torch


class CAG(MTLDOGALGO):
    def __init__(self):
        super().__init__()

    def backward(self, losses: Dict[str, Tensor]):
        
        share_grads = torch.stack([self.get_grads_share(loss, mode='backward') for loss in losses.values()])
        share_grads = torch.mean(share_grads, dim=1)
        mean_share_grads = self.cagrad(share_grads, num_tasks=len(losses))
        
        head_grad = []
        for loss in losses.values():
            head_grad.append(self.get_grads_heads(loss, mode='backward'))
        mean_head_grad = {head: torch.mean(torch.stack([grad[head] for grad in head_grad]), dim=0).to(device=self.device) for head in self.args.tkss}
        
        self.reset_grad_share(mean_share_grads)
        self.reset_grad_heads(mean_head_grad)
        
        return None, None
    
    def cagrad(self, grad_vec, num_tasks):
        """
        grad_vec: [num_tasks, dim]
        """
        grads = grad_vec

        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG)+1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True)
        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c = (gg+1e-4).sqrt() * self.args.cagradc
        
        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward()
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww)+1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm+1e-4)
        g = ((1/num_tasks + ww * lmbda).view(-1, 1).to(grads.device) * grads).sum(0) / (1 + self.args.cagradc**2)
        return g
    

def algo_cag():
    return CAG