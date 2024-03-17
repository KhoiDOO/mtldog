from argparse import Namespace
from .core import MTLDOGTR
from typing import Dict
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from alive_progress import alive_it
from statistics import mean

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

        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(gpu)

        tr_loaders = []
        te_loaders = []

        for idx, (trds, teds) in enumerate(zip(self.tr_dss, self.te_dss)):
            tr_sampler = DistributedSampler(trds)
            te_sampler = DistributedSampler(teds)
            per_device_bs = self.args.bs // self.args.world_size

            trl = DataLoader(dataset=trds, batch_size=per_device_bs, num_workers=self.args.wk, pin_memory=self.args.pm, sampler=tr_sampler)
            tel = DataLoader(dataset=teds, batch_size=per_device_bs, num_workers=self.args.wk, pin_memory=self.args.pm, sampler=te_sampler)

            if idx in self.args.trdms:
                tr_loaders.append(trl)
                te_loaders.append(tel)
            else:
                te_loaders.append(trl)
                te_loaders.append(tel)

        class Agent(self.model, self.algo):
            def __init__(self, args):
                super().__init__(args)
        
        agent = Agent(args=args).cuda(gpu)
        agent = DDP(agent, device_ids=[gpu])
        optimizer = Adam(params=agent.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)

        if args.rank == 0:
            bar = alive_it(range(args.epoch))
        else:
            bar = range(args.epoch)

        for epoch in bar:
            agent.train()
            
            for trld in tr_loaders:
                trld.sampler.set_epoch(epoch)
                trdm_idx = trld.dataset.domain
                trdm_txt = trld.dataset.idx2dm[trdm_idx]
                
                for (input, target) in trld:
                    
                    losses = torch.zeros(len(args.tkss)).cuda(gpu)
                    
                    input: Tensor = input.cuda(gpu)
                    target: Dict[str, Tensor] = {tk: target[tk].cuda(gpu) for tk in target}
                    output: Dict[str, Tensor] = agent(input)

                    for tkix, tk in enumerate(output):
                        for loss_key in self.loss_dct:
                            if tk in loss_key:
                                losses[tkix] = self.loss_dct[loss_key](output[tk], target[tk])

                                if args.rank == 0:
                                    trdm_loss_key = f"{trdm_txt}/train-in-{tk}-{loss_key.split('_')[-1]}"
                                    self.track(trdm_loss_key, losses[tkix].item())
                        
                        if args.rank == 0:
                            for metric_key in self.metric_dct:
                                if tk in metric_key:
                                    trdm_metric_key = f"{trdm_txt}/train-in-{tk}-{metric_key.split('_')[-1]}"
                                    self.track(trdm_metric_key, self.metric_dct[metric_key](output[tk], target[tk]))

                    
                    optimizer.zero_grad()
                    sol_grad = agent.module.backward(losses=losses)
                    optimizer.step()
            
            if args.rank == 0:
                if args.wandb:
                    self.sync(epoch=epoch)
                else:
                    self.show_log_dict()
            
            agent.eval()
            if args.rank == 0:
                with torch.no_grad():

                    for teld in te_loaders:
                        teld.sampler.set_epoch(epoch)
                        tedm_idx = teld.dataset.domain
                        tedm_txt = teld.dataset.idx2dm[tedm_idx]

                        for (input, target) in teld:
                            losses = torch.zeros(len(args.tkss)).cuda(gpu)
                            
                            input: Tensor = input.cuda(gpu)
                            target: Dict[str, Tensor] = {tk: target[tk].cuda(gpu) for tk in target}
                            output: Dict[str, Tensor] = agent(input)

                            for tkix, tk in enumerate(output):
                                for loss_key in self.loss_dct:
                                    if tk in loss_key:
                                        losses[tkix] = self.loss_dct[loss_key](output[tk], target[tk])

                                        train_txt = 'train' if teld.dataset.tr is True else 'test'
                                        inout_txt = 'in' if tedm_idx in args.trdms else 'out'

                                        tedm_loss_key = f"{tedm_txt}/{train_txt}-{inout_txt}-{tk}-{loss_key.split('_')[-1]}"
                                        self.track(tedm_loss_key, losses[tkix].item())

                                for metric_key in self.metric_dct:
                                    if tk in metric_key:
                                        
                                        train_txt = 'train' if teld.dataset.tr is True else 'test'
                                        inout_txt = 'in' if tedm_idx in args.trdms else 'out'
                                        
                                        tedm_metric_key = f"{tedm_txt}/{train_txt}-{inout_txt}-{tk}-{metric_key.split('_')[-1]}"
                                        
                                        self.track(key=tedm_metric_key, value=self.metric_dct[metric_key](output[tk], target[tk]))
                
                if args.wandb:
                    self.sync(epoch=epoch)
            
            scheduler.step()
        self.cleanup()