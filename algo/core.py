from typing import List
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

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad
    
    def _compute_grad(self, losses: Tensor, mode: str):
        grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        for tn in range(self.task_num):
            if mode == 'backward':
                losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                grads[tn] = self._grad2vec()
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                grads[tn] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError('No support {} mode for gradient computation')
            self.zero_grad_share_params()
        return grads
    
    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
        
    def _get_grads(self, losses: Tensor, mode: str='backward'):
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode)
        return grads
    
    def _backward_new_grads(self, batch_weight, grads=None):
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
        self._reset_grad(new_grads)
    

    def backward(self, losses: List, args: Namespace):
        raise NotImplementedError()

    @property
    def train_loss_buffer(self):
        return self.train_loss_buffer

    @property
    def params(self):
        return self.params