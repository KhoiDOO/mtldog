from argparse import Namespace
from .core import MTLDOGTR
from typing import Dict
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from alive_progress import alive_it

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

class SUP(MTLDOGTR):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    def run(self):
        mp.spawn(self.train, (self.args,), self.args.ngpus_per_node)

    def train(self, gpu, args):
        args.rank += gpu

        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(gpu)

        agent = DDP(self.agent, device_ids=[gpu])
        optimizer = Adam(params=agent.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        if args.rank == 0:
            bar = alive_it(range(args.epochs))
        else:
            bar = range(args.epochs)

        for epoch in bar:
            self.agent.train()
            trdm_loss_dct = {}
            trdm_metric_dct = {}
            
            for trld in self.tr_loaders:
                trld.sampler.set_epoch(epoch)
                trdm_idx = trld.dataset.domain()
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
                                    if trdm_loss_key not in trdm_loss_dct:
                                        trdm_loss_dct[trdm_loss_key] = losses[tkix].item()
                                    else:
                                        trdm_loss_dct[trdm_loss_key] += losses[tkix].item()
                        if args.rank == 0:
                            for metric_key in self.metric_dct:
                                if tk in metric_key:
                                    trdm_metric_key = f"{trdm_txt}/train-in-{tk}-{metric_key.split('_')[-1]}"

                                    if trdm_metric_key not in trdm_metric_dct:
                                        trdm_metric_dct[trdm_metric_key] = self.metric_dct[metric_key](output[tk], target[tk])
                                    else:
                                        trdm_metric_dct[trdm_metric_key] += self.metric_dct[metric_key](output[tk], target[tk])

                    
                    optimizer.zero_grad()
                    sol_grad = agent.backward(losses=losses)
                    optimizer.step()
            
            if args.rank == 0:
                mean_trdm_loss_dct = {trdm_key : trdm_loss_dct[trdm_key]/len(self.tr_loaders) for trdm_key in trdm_loss_dct}
                mean_trdm_metric_dct = {trdm_key : trdm_metric_dct[trdm_key]/len(self.tr_loaders) for trdm_key in trdm_metric_dct}

                for trdm_key in mean_trdm_loss_dct:
                    self.__run.log({trdm_key: mean_trdm_loss_dct[trdm_key], 'epoch': epoch})
                for trdm_key in mean_trdm_metric_dct:   
                    self.__run.log({trdm_key: mean_trdm_metric_dct[trdm_key], 'epoch': epoch})
            
            self.agent.eval()
            if args.rank == 0:
                with torch.no_grad():
                    tedm_loss_dct = {}
                    tedm_metric_dct = {}

                    for teld in self.te_loaders:
                        teld.sampler.set_epoch(epoch)
                        tedm_idx = teld.dataset.domain()
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

                                        train_txt = 'train' if teld.dataset.train is True else 'test'
                                        inout_txt = 'in' if tedm_idx in args.trdms else 'out'

                                        tedm_loss_key = f"{tedm_txt}/{train_txt}-{inout_txt}-{tk}-{loss_key.split('_')[-1]}"
                                        if tedm_loss_key not in tedm_loss_dct:
                                            tedm_loss_dct[tedm_loss_key] = losses[tkix].item()
                                        else:
                                            tedm_loss_dct[tedm_loss_key] += losses[tkix].item()

                                for metric_key in self.metric_dct:
                                    if tk in metric_key:
                                        
                                        train_txt = 'train' if teld.dataset.train is True else 'test'
                                        inout_txt = 'in' if tedm_idx in args.trdms else 'out'
                                        
                                        tedm_metric_key = f"{tedm_txt}/{train_txt}-{inout_txt}-{tk}-{metric_key.split('_')[-1]}"
                                        if tedm_metric_key not in tedm_metric_dct:
                                            tedm_metric_dct[tedm_metric_key] = self.metric_dct[metric_key](output[tk], target[tk])
                                        else:
                                            tedm_metric_dct[tedm_metric_key] += self.metric_dct[metric_key](output[tk], target[tk])
                
                mean_tedm_loss_dct = {tedm_key : tedm_loss_dct[tedm_key]/len(self.tr_loaders) for tedm_key in tedm_loss_dct}
                mean_tedm_metric_dct = {tedm_key : tedm_metric_dct[tedm_key]/len(self.tr_loaders) for tedm_key in tedm_metric_dct}

                for tedm_key in mean_tedm_loss_dct:
                    self.__run.log({tedm_key: mean_tedm_loss_dct[tedm_key], 'epoch': epoch})
                for tedm_key in mean_tedm_metric_dct:
                    self.__run.log({tedm_key: mean_tedm_metric_dct[tedm_key], 'epoch': epoch})
            
            scheduler.step()
        self.cleanup()