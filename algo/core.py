from typing import Dict, List, Tuple
from argparse import Namespace
from torch import nn, Tensor

import json
import torch
import numpy as np

class MTLDOGALGO(nn.Module):
    def __init__(self):
        super(MTLDOGALGO, self).__init__()

    def init_param(self, args: Namespace) -> None:
        self.args: Namespace = args
        self.tkss: List = args.tkss
        self.train_loss_buffer = np.zeros([args.task_num, args.round])
        self.hparams_path = args.hp
        self.params = json.load(open(self.hparams_path, 'r'))

    # Extract ==================================================================================================================
    def compute_grad_dim_share(self) -> None:
        self.grad_index_share : List[int] = []
        for param in self.get_share_params():
            self.grad_index_share.append(param.data.numel())
        self.grad_dim_share : int = sum(self.grad_index)
    
    def compute_grad_dim_heads(self) -> None:
        self.grad_index_heads : Dict[str, List[int]] = {}
        head_params_dict : Dict[str, List[Tensor]] = self.get_heads_params()
        
        for head in head_params_dict:
            for param in head_params_dict[head]:
                if head in self.grad_index_heads:
                    self.grad_index_heads[head] = [param.data.numel()]
                else:
                    self.grad_index_heads[head].append(param.data.numel())
        
        self.grad_dim_heads : Dict[str, List[int]] = {head : sum(self.grad_index_heads[head]) for head in self.grad_index_heads}

    def grad2vec_share(self) -> Tensor:
        grad = torch.zeros(self.grad_dim_share)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index_share[:count])
                end = sum(self.grad_index_share[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def grad2vec_heads(self, head:str) -> Tensor:
        grad = torch.zeros(self.grad_dim_heads[head])
        for param in self.get_heads_params()[head]:
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index_heads[head][:count])
                end = sum(self.grad_index_heads[head][:(count+1)])
                grad[head][beg:end] = param.grad.data.view(-1)
            count += 1
        return grad
    
    def compute_grad_share(self, losses: Tensor, mode: str) -> Tensor:
        grads = torch.zeros(self.task_num, self.grad_dim_share).to(self.device)
        
        for tkidx in range(self.task_num):
            if mode == 'backward':
                losses[tkidx].backward(retain_graph=True)
                grads[tkidx] = self.grad2vec_share()
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(losses[tkidx], self.get_share_params(), retain_graph=True))
                grads[tkidx] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError(f'No support {mode} mode for gradient computation')
            
            self.zero_grad_share_params()
        return grads
    
    def compute_grad_head(self, losses: Tensor, mode: str) -> Tensor:
        grads = {head : torch.zeros(self.grad_dim_heads[head]) for head in self.grad_dim_heads}

        for tkidx, tk in enumerate(grads):
            if mode == 'backward':
                losses[tkidx].backward(retain_graph=True)
                grads[tk] = self.grad2vec_heads(head=tk)
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(losses[tkidx], self.get_heads_params(), retain_graph=True))
                grads[tk] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError(f'No support {mode} mode for gradient computation')
        
        self.zero_grad_heads_params()
        return grads
        
    def get_grads_share(self, losses: Tensor, mode: str='backward') -> Tensor:
        self.compute_grad_dim_share()
        grads = self.compute_grad_share(losses, mode)
        return grads
    
    def get_grads_heads(self, losses: Tensor, mode: str='backward') -> Dict[str, Tensor]:
        self.compute_grad_dim_heads()
        grads = self.compute_grad_head(losses, mode)
        return grads

    def get_grads_share_heads(self, losses: Tensor, mode: str='backward') -> Tuple[Tensor, Dict[str, Tensor]]:
        self.compute_grad_dim_share()
        self.compute_grad_dim_heads()

        share_grads = torch.zeros(self.task_num, self.grad_dim_share).to(self.device)
        heads_grads = {head : torch.zeros(self.grad_dim_heads[head]) for head in self.grad_dim_heads}

        for tkidx, tk in enumerate(self.tkss):
            if mode == 'backward':
                losses[tkidx].backward(retain_graph=True)
                share_grads[tkidx] = self.grad2vec_share()
                heads_grads[tk] = self.grad2vec_heads(head=tk)
            elif mode == 'autograd':
                share_grad = list(torch.autograd.grad(losses[tkidx], self.get_share_params(), retain_graph=True))
                share_grads[tkidx] = torch.cat([g.view(-1) for g in share_grad])
                head_grad = list(torch.autograd.grad(losses[tkidx], self.get_heads_params(), retain_graph=True))
                heads_grads[tk] = torch.cat([g.view(-1) for g in head_grad])
            else:
                raise ValueError(f'No support {mode} mode for gradient computation')
        
        self.zero_grad_share_params()
        self.zero_grad_heads_params()
        return share_grads, heads_grads

    # Extract ==================================================================================================================

    # Update ==================================================================================================================
    def reset_grad_share(self, new_grads: Tensor):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index_share[:count])
                end = sum(self.grad_index_share[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
    
    def reset_grad_heads(self, new_grads : Dict[str, Tensor]):
        count = {tk : 0 for tk in self.tkss}

        for head, params in self.get_heads_params():
            for param in params:
                if param.grad is not None:
                    beg = 0 if count == 0 else sum(self.grad_index_heads[head][:count[head]])
                    end = sum(self.grad_index_heads[head][:(count[head]+1)])
                    param.grad.data = new_grads[head][beg:end].contiguous().view(param.data.size()).data.clone()
                count[head] += 1
    
    def backward_new_grads_share(self, batch_weight, grads=None):
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
        self.reset_grad_share(new_grads)

    def backward(self, losses: Dict[str, Tensor]):
        raise NotImplementedError()
    # Update ==================================================================================================================