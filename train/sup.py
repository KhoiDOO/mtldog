from argparse import Namespace
from .core import MTLDOGTR
from typing import Dict, List, Tuple
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from ds import DistributedInfiniteDataLoader, SmartDistributedSampler
from alive_progress import alive_it

import os, sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

class SUP(MTLDOGTR):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    def run(self):
        mp.spawn(self.train, (self.args,), self.args.ngpus)

    def train(self, gpu, args):
        args.rank += gpu

        f = open(os.devnull, "w")
        if args.rank != 0:
            sys.stdout = f
            sys.stderr = f

        print("Initializing Process Group...")
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(gpu)

        tr_loaders = []
        te_loaders = []
        for idx, (trds, teds) in enumerate(zip(self.tr_dss, self.te_dss)):
            tr_sampler = SmartDistributedSampler(trds)
            per_device_bs = args.bs // args.world_size

            trl = DistributedInfiniteDataLoader(dataset=trds, batch_size=per_device_bs, num_workers=self.args.wk, pin_memory=self.args.pm, sampler=tr_sampler)
            tel = DataLoader(dataset=teds, batch_size=per_device_bs, num_workers=self.args.wk, pin_memory=self.args.pm)

            if idx in self.args.trdms:
                tr_loaders.append(trl)
                te_loaders.append(tel)
            else:
                te_loaders.append(trl)
                te_loaders.append(tel)
        
        class Agent(self.model, self.algo):
            def __init__(self, args):
                super().__init__(args)

                self.device = gpu
                self.init_param(args)
        
        agent = Agent(args=args).cuda(gpu)
        agent = DDP(agent, device_ids=[gpu])
        optimizer = Adam(params=agent.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.round)

        if args.rank == 0:
            bar = alive_it(range(args.round), length = 80)
        else:
            bar = range(args.round)
        
        trdm_txts = [trld.dataset.domain_txt for trld in tr_loaders]

        for round in bar:

            trdm_batchs = next(zip(*tr_loaders))
            agent.train()
            for trld in tr_loaders:
                trld.sampler.set_round(round)
            
            losses = {dmtxt : torch.zeros(len(args.tkss)).cuda(gpu) for dmtxt in trdm_txts}
            
            for trdmb, trdm_txt in zip(trdm_batchs, trdm_txts):
                input, target = trdmb
                input: Tensor = input.cuda(gpu)
                target: Dict[str, Tensor] = {tk: target[tk].cuda(gpu) for tk in target}
                output: Dict[str, Tensor] = agent(input)

                for tkix, tk in enumerate(args.tkss):
                    for loss_key in self.loss_dct:
                        if tk in loss_key:
                            losses[trdm_txt][tkix] = self.loss_dct[loss_key](output[tk], target[tk], args)

                            if args.rank == 0:
                                trdm_loss_key = f"{trdm_txt}/train-in-{tk}-{loss_key.split('_')[-1]}"
                                self.track(trdm_loss_key, losses[trdm_txt][tkix].item())
            
            if args.ramk == 0 and args.log_grad:
                grad_dict = {}
                for dmtxt in trdm_txts:
                    grad_share, grad_heads = agent.module.get_grads_share_heads(losses = losses[dmtxt])
                    grad_dict[dmtxt] = {
                        'share' : grad_share.detach().clone().cpu(), 
                        'heads' : {head : grad_heads[head].detach().clone().cpu() for head in grad_heads}}

                    
            optimizer.zero_grad()
            sol_grad_share, sol_grad_head = agent.module.backward(losses=losses)
            optimizer.step()

            if args.rank == 0:
                agent.eval()
                for trdmb, trdm_txt in zip(trdm_batchs, trdm_txts):
                    input, target = trdmb
                    input: Tensor = input.cuda(gpu)
                    target: Dict[str, Tensor] = {tk: target[tk].cuda(gpu) for tk in target}
                    output: Dict[str, Tensor] = agent(input)
                    
                    for tk in args.tkss:
                        for metric_key in self.metric_dct:
                            if tk in metric_key:
                                trdm_metric_key = f"{trdm_txt}/train-in-{tk}-{metric_key.split('_')[-1]}"
                                self.track(trdm_metric_key, self.metric_dct[metric_key](output[tk], target[tk]))
                
                if args.wandb:
                    self.sync()
                else:
                    self.show_log(round=round, stage='TRAINING')
                
                if args.log_grad:
                    if args.wandb:
                        self.track_sync_grad_train(grad_dict=grad_dict, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head)
            
            agent.eval()
            if args.rank == 0 and round % args.chkfreq == 0 and round != 0:
                tedm_txts = [teld.dataset.domain_txt for teld in te_loaders]
                train_txts = ['train' if teld.dataset.tr is True else 'test' for teld in te_loaders]
                inout_txts = ['in' if teld.dataset.domain_idx in args.trdms else 'out' for teld in te_loaders]

                for teld, tedm_txt, train_txt, inout_txt in zip(te_loaders, tedm_txts, train_txts, inout_txts):
                    for (input, target) in teld:

                        __losses = torch.zeros(len(args.tkss)).cuda(gpu)

                        input: Tensor = input.cuda(gpu)
                        target: Dict[str, Tensor] = {tk: target[tk].cuda(gpu) for tk in target}
                        output: Dict[str, Tensor] = agent(input)

                        for tkix, tk in enumerate(args.tkss):
                            for loss_key in self.loss_dct:
                                if tk in loss_key:
                                    __losses[tkix] = self.loss_dct[loss_key](output[tk], target[tk], args)
                                    tedm_loss_key = f"{tedm_txt}/{train_txt}-{inout_txt}-{tk}-{loss_key.split('_')[-1]}"
                                    self.track(tedm_loss_key, __losses[tkix].item())

                            for metric_key in self.metric_dct:
                                if tk in metric_key:
                                    tedm_metric_key = f"{tedm_txt}/{train_txt}-{inout_txt}-{tk}-{metric_key.split('_')[-1]}"
                                    self.track(key=tedm_metric_key, value=self.metric_dct[metric_key](output[tk], target[tk]))
                
                if args.wandb:
                    self.sync()
                else:
                    self.show_log(round=round, stage='EVALUATION')
            
            scheduler.step()
        self.finish()