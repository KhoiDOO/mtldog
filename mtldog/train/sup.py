from argparse import Namespace
from .core import MTLDOGTR
from typing import Dict
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from ds import DistributedInfiniteDataLoader, SmartDistributedSampler
from alive_progress import alive_it
from statistics import mean

from arch import MTLDOGARCH
from algo import MTLDOGALGO

import os, sys, torch
import torch.multiprocessing as mp
import torch.distributed as dist

class SUP(MTLDOGTR):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    def run(self):
        mp.spawn(self.train, (self.args,), self.args.ngpus)

    def train(self, gpu, args):
        args.rank += gpu
        is_master = args.rank == 0

        f = open(os.devnull, "w")
        if args.rank != 0:
            sys.stdout = f
            sys.stderr = f

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
            def __init__(self, args) -> MTLDOGARCH | MTLDOGALGO:
                super(Agent, self).__init__(args)

                self.device = gpu
                self.init_param(args)
        
        agent = Agent(args=args).cuda(gpu)
        agent = DDP(agent, device_ids=[gpu])
        optimizer = Adam(params=agent.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.round)
        
        bar = alive_it(range(args.round), length = 80) if is_master else range(args.round)
        trdm_txts = [trld.dataset.domain_txt for trld in tr_loaders]
        old_eval_loss = 1e26
        remap = False

        for round in bar:
            checkpoint = (round + 1) % args.chkfreq == 0
            
            trdm_batchs = next(zip(*tr_loaders))
            agent.train()
            for trld in tr_loaders:
                trld.sampler.set_round(round)
            
            train_losses = {dmtxt : torch.zeros(len(args.tkss)).cuda(gpu) for dmtxt in trdm_txts}
            
            for trdmb, trdm_txt in zip(trdm_batchs, trdm_txts):
                input, target = trdmb
                input: Tensor = input.cuda(gpu)
                target: Dict[str, Tensor] = {tk: target[tk].cuda(gpu) for tk in target}
                output: Dict[str, Tensor] = agent(input)

                for tkix, tk in enumerate(args.tkss):
                    for loss_key in self.loss_dct:
                        if tk in loss_key:
                            train_losses[trdm_txt][tkix] = self.loss_dct[loss_key](output[tk], target[tk], args)

                            if is_master:
                                trdm_loss_key = f"{trdm_txt}/train-in-{tk}-{loss_key.split('_')[-1]}"
                                self.track(trdm_loss_key, train_losses[trdm_txt][tkix].item())
            
            if is_master:
                if args.grad:
                    grad_dict = agent.module.get_grads_dm_share_heads(losses = train_losses, detach = True)
                if args.hess:
                    hess_dict = agent.module.get_grads_hess_dm_share_heads(losses = train_losses)
                    
            optimizer.zero_grad()
            sol_grad_share, sol_grad_head = agent.module.backward(losses=train_losses)
            optimizer.step()

            if is_master:
                agent.eval()
                with torch.no_grad():
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
            
            agent.eval()
            with torch.no_grad():
                if is_master and not args.diseval and checkpoint:
                    tedm_txts = [teld.dataset.domain_txt for teld in te_loaders]
                    train_txts = ['train' if teld.dataset.tr is True else 'test' for teld in te_loaders]
                    inout_txts = ['in' if teld.dataset.domain_idx in args.trdms else 'out' for teld in te_loaders]
                    eval_loss_lst = []

                    for teld, tedm_txt, train_txt, inout_txt in zip(te_loaders, tedm_txts, train_txts, inout_txts):
                        for (input, target) in teld:

                            eval_losses = torch.zeros(len(args.tkss)).cuda(gpu)

                            input: Tensor = input.cuda(gpu)
                            target: Dict[str, Tensor] = {tk: target[tk].cuda(gpu) for tk in target}
                            output: Dict[str, Tensor] = agent(input)

                            for tkix, tk in enumerate(args.tkss):
                                for loss_key in self.loss_dct:
                                    if tk in loss_key:
                                        eval_losses[tkix] = self.loss_dct[loss_key](output[tk], target[tk], args)
                                        tedm_loss_key = f"{tedm_txt}/{train_txt}-{inout_txt}-{tk}-{loss_key.split('_')[-1]}"
                                        self.track(tedm_loss_key, eval_losses[tkix].item())

                                for metric_key in self.metric_dct:
                                    if tk in metric_key:
                                        tedm_metric_key = f"{tedm_txt}/{train_txt}-{inout_txt}-{tk}-{metric_key.split('_')[-1]}"
                                        self.track(key=tedm_metric_key, value=self.metric_dct[metric_key](output[tk], target[tk]))
                            
                            eval_loss_lst.append(torch.sum(eval_losses).item())
            
            if is_master and checkpoint:
                save_dict = {
                    'args' : args,
                    'model_state_dict': agent.module.state_dict()
                }
                
                if not args.diseval:
                    mean_loss = mean(eval_loss_lst)
                else:
                    mean_loss = mean([torch.sum(item).item() for _, item in train_losses.items()])
            
                if mean_loss < old_eval_loss:
                    torch.save(save_dict, self.best_model_path)
                    remap = True
                else:
                    remap = False
                torch.save(save_dict, self.last_model_path)
            
            if is_master:
                self.sync(grad_dict=grad_dict if args.grad else None, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head, hess_dict=hess_dict if args.hess else None)
            
            dist.barrier()
            if checkpoint:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % args.rank}
                if remap:
                    agent.module.load_state_dict(torch.load(self.best_model_path, map_location=map_location)['model_state_dict'])

            scheduler.step()
        
        if is_master and args.wandb:
            self.sync_finish()
        
        self.finish()