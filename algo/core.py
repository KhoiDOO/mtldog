from typing import Dict, List
from argparse import Namespace
from torch import nn, Tensor

import json
import torch
import numpy as np

class MTLDOGALGO(nn.Module):
    def __init__(self):
        super(MTLDOGALGO, self).__init__()

    def init_param(self, args: Namespace):
        self.train_loss_buffer = np.zeros([args.task_num, args.epochs])
        self.hparams_path = args.hp
        self.params = json.load(open(self.hparams_path, 'r'))

    def compute_grad_dim_share(self):
        self.grad_index_share : List[int] = []
        for param in self.get_share_params():
            self.grad_index_share.append(param.data.numel())
        self.grad_dim_share : int = sum(self.grad_index)
    
    def compute_grad_dim_heads(self):
        self.grad_index_heads : Dict[str, List[int]] = {}
        head_params_dict : Dict[str, List[Tensor]] = self.get_heads_params()
        
        for head in head_params_dict:
            for param in head_params_dict[head]:
                if head in self.grad_index_heads:
                    self.grad_index_heads[head] = [param.data.numel()]
                else:
                    self.grad_index_heads[head].append(param.data.numel())
        
        self.grad_dim_heads : Dict[str, List[int]] = {head : sum(self.grad_index_heads[head]) for head in self.grad_index_heads}

    def grad2vec_share(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad
    
    def compute_grad_share(self, losses: Tensor, mode: str):
        grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        for tn in range(self.task_num):
            if mode == 'backward':
                losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                grads[tn] = self.grad2vec_share()
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                grads[tn] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError('No support {} mode for gradient computation')
            self.zero_grad_share_params()
        return grads
    
    def reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
        
    def get_grads(self, losses: Tensor, mode: str='backward'):
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode)
        return grads
    
    def backward_new_grads(self, batch_weight, grads=None):
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
        self._reset_grad(new_grads)
    

    def backward(self, losses: Dict[str, List[Tensor]]):
        raise NotImplementedError()

    @property
    def train_loss_buffer(self):
        return self.train_loss_buffer

    @property
    def params(self):
        return self.params