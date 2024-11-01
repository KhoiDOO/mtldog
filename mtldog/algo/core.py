from typing import Dict, List, Tuple
from argparse import Namespace
from torch import nn, Tensor
from arch import MTLDOGARCH

import json
import torch
import numpy as np

class MTLDOGALGO(nn.Module):
    def __init__(self) -> MTLDOGARCH:
        super(MTLDOGALGO, self).__init__()

    def init_param(self, args: Namespace) -> None:
        self.args: Namespace = args
        self.spcnt = args.spcnt
        self.tkss: List = args.tkss
        self.trdms: List = args.trdms
        self.train_loss_buffer = np.zeros([args.task_num, args.round])
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed)

    # Extract ==================================================================================================================
    def compute_grad_dim_share(self) -> None:
        self.grad_index_share : List[int] = []
        for param in self.get_share_params():
            self.grad_index_share.append(param.data.numel())
        self.grad_dim_share : int = sum(self.grad_index_share)
    
    def compute_grad_dim_heads(self) -> None:
        self.grad_index_heads : Dict[str, List[int]] = {}
        head_params_dict : Dict[str, List[Tensor]] = self.get_heads_params()
        
        for head in head_params_dict:
            for param in head_params_dict[head]:
                if head not in self.grad_index_heads:
                    self.grad_index_heads[head] = [param.data.numel()]
                else:
                    self.grad_index_heads[head].append(param.data.numel())
        
        self.grad_dim_heads : Dict[str, List[int]] = {head : sum(self.grad_index_heads[head]) for head in self.args.tkss}

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
        count = 0
        for param in self.get_heads_params()[head]:
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index_heads[head][:count])
                end = sum(self.grad_index_heads[head][:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
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
        heads_grads = {head : torch.zeros(self.grad_dim_heads[head]) for head in self.args.tkss}

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

    def get_grads_dm_share_heads(self, losses: Dict[str, Tensor], detach: bool) -> Dict[str, Tensor | Dict[str, Tensor]]:
        grad_dict = {}
        for dmtxt in losses:
            grad_share, grad_heads = self.get_grads_share_heads(losses = losses[dmtxt])
            grad_dict[dmtxt] = {
                'share' : grad_share.detach().clone().cpu() if detach else grad_share, 
                'heads' : {head : grad_heads[head].detach().clone().cpu() if detach else grad_heads[head] for head in grad_heads}}
        
        return grad_dict
    
    def hess_approx_gauss_newton_barlett(self, grads) -> List[Tensor]:
        return [x * x for x in grads]

    def hess_approx_hutchinson(self, params, grads) -> List[Tensor]:

        hess = [0] * len(grads)

        for i in range(self.spcnt):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=True)
            for idx, (h_z, z) in enumerate(zip(h_zs, zs)):
                hess[idx] += h_z * z / self.spcnt  # approximate the expected values of z*(H@z)
        
        return hess
    
    def get_grads_hess_dm_share_heads(self, losses: Dict[str, Tensor]):

        hess_dict = {}

        for dm in losses:
            task_losses = losses[dm]

            task_dict = {tk : None for tk in self.tkss}

            for tkidx, tk in enumerate(self.tkss):
                task_losses[tkidx].backward(retain_graph=True)

                name_share = self.name_share_params_require_grad()
                name_heads = self.name_heads_params_require_grad()[tk]

                share_params = self.get_share_params_require_grad()
                heads_params = self.get_heads_params_require_grad()[tk]
        
                temp_share_grad = list(torch.autograd.grad(task_losses[tkidx], self.get_share_params(), retain_graph=True, create_graph=True))
                temp_heads_grad = list(torch.autograd.grad(task_losses[tkidx], self.get_heads_params()[tk], retain_graph=True, create_graph=True))
                
                temp_share_hess = self.hess_approx_hutchinson(share_params, temp_share_grad)
                temp_heads_hess = self.hess_approx_hutchinson(heads_params, temp_heads_grad)

                temp_share_grad = [g.detach().clone().cpu() for g in temp_share_grad]
                temp_heads_grad = [g.detach().clone().cpu() for g in temp_heads_grad]

                temp_share_hess = [h.detach().clone().cpu() for h in temp_share_hess]
                temp_heads_hess = [h.detach().clone().cpu() for h in temp_heads_hess]

                share_grad_hess_dict = {'name':name_share, 'grad':temp_share_grad, 'hess':temp_share_hess}
                head_grad_hess_dict = {'name':name_heads, 'grad':temp_heads_grad, 'hess':temp_heads_hess}

                task_dict[tk] = {'share':share_grad_hess_dict, 'head':head_grad_hess_dict}

                # self.zero_grad_share_params()
                # self.zero_grad_heads_params()
            
            hess_dict[dm] = task_dict
        
        return hess_dict

        """
        {
            "dmtxt" : {
                "tk" {
                    "share" : {'name' : [Tensor], 'grad' : [Tensor], 'hess' : [Tensor]},
                    "heads" : {'name' : [Tensor], 'grad' : [Tensor], 'hess' : [Tensor]}
                }
            }
        }
        """


    # Extract ==================================================================================================================

    # Update ===================================================================================================================
    def reset_grad_share(self, new_grads: Tensor):
        count = 0
        for param in self.get_share_params():
            beg = 0 if count == 0 else sum(self.grad_index_share[:count])
            end = sum(self.grad_index_share[:(count+1)])
            param.grad = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
    
    def reset_grad_heads(self, new_grads : Dict[str, Tensor]):
        count = {tk : 0 for tk in self.tkss}

        for head, params in self.get_heads_params().items():
            for param in params:
                beg = 0 if count == 0 else sum(self.grad_index_heads[head][:count[head]])
                end = sum(self.grad_index_heads[head][:(count[head]+1)])
                param.grad = new_grads[head][beg:end].contiguous().view(param.data.size()).data.clone()
                count[head] += 1
    
    def backward_new_grads_share(self, batch_weight, grads=None):
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
        self.reset_grad_share(new_grads)

    def backward(self, losses: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        
        """
        losses = {
            "domain_0" : Tensor[<task_loss_0>, <task_loss_1>, ..., <task_loss_m>],
            ...,
            "domain_n" : Tensor[<task_loss_0>, <task_loss_1>, ..., <task_loss_m>],
        }
        """

        
        raise NotImplementedError()
        
    # Update ===================================================================================================================