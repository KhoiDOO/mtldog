import os, sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import torch
from typing import *
from argparse import Namespace

import wandb
import random
import numpy as np
import json
import hashlib

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import ds
import arch
import algo

class MTLDOGTR:
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.prepare_seed()
        self.prepare_device()
        self.prepare_ds()
        self.prepare_save_dir()
        self.prepare_algo()
        self.prepare_wandb()

    def prepare_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.gen_func = torch.Generator().manual_seed(self.args.seed)
    
    def prepare_device(self):
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.args.dvids)

            self.args.ngpus = torch.cuda.device_count()
            self.args.rank = 0
            self.args.dist_url = f'tcp://localhost:{self.args.port}'
            self.args.world_size = self.args.ngpus
        else:
            raise RuntimeError("CUDA is not available")
    
    def prepare_ds(self):
        ds_map = vars(ds)
        self.ds_dct = {k : ds_map[k] for k in ds_map if 'ds' in k}

        self.args, tr_dss, te_dss = self.ds_dct[f'ds_{self.args.ds}'](self.args)

        assert self.args.bs % self.args.world_size == 0

        self.tr_loaders = []
        self.te_loaders = []

        for trds, teds in zip(tr_dss, te_dss):
            tr_sampler = DistributedSampler(trds)
            te_sampler = DistributedSampler(teds)
            per_device_bs = self.args.bs // self.args.world_size

            trl = DataLoader(dataset=trds, batch_size=per_device_bs, num_workers=self.args.wk, pin_memory=self.args.pm, sampler=tr_sampler, generator=self.gen_func)
            tel = DataLoader(dataset=teds, batch_size=per_device_bs, num_workers=self.args.wk, pin_memory=self.args.pm, sampler=te_sampler, generator=self.gen_func)

            self.tr_loaders.append(trl)
            self.te_loaders.append(tel)

    def prepare_save_dir(self):
        main_dir = os.getcwd()

        run_dir = main_dir + '/runs'
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        
        self.method_dir = run_dir + f'/{self.args.method}'
        if not os.path.exists(self.method_dir):
            os.mkdir(self.method_dir)
        
        self.ds_dir = self.method_dir + f'/{self.args.ds}'
        if not os.path.exists(self.ds_dir):
            os.mkdir(self.ds_dir)

        self.args.hashcode = self.get_hash()
        self.save_dir = self.ds_dir + f"/{self.args.hashcode}"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.args.save_dir = self.save_dir
    

    def prepare_wandb(self):
        self.args.run_name = f'{self.args.method}__{self.args.ds}__{self.args.hashcode}'

        self.__run = wandb.init(
            project=self.args.wandb_prj,
            entity=self.args.wandb_entity,
            config=self.args,
            name=self.args.run_name,
            force=True
        )
    
    def prepare_algo(self):
        algo_map = vars(algo)
        self.algo_dct = {k : algo_map[k] for k in algo_map if 'algo' in k}
        self.algo = self.algo_dct[f'algo_{self.args.m}']()

        model_map = vars(arch)
        self.model_dct = {k : model_map[k] for  k in model_map if 'model' in k}
        self.model = self.model_dct[f'model_{self.args.model}']()

        class Agent(self.model, self.algo):
            def __init__(self, args):
                super().__init__(args)
        
        self.agent = Agent(self.args)
    
    def log_wbmodel(self):
        best_path = self.args.save_dir + f'/best.pt'
        if os.path.exists(best_path):
            self.__run.log_model(path=best_path, name=f'{self.args.run_name}-best-model')
        else:
            raise Exception(f'best model path is not exist at {best_path}')
        
        last_path = self.args.save_dir + f'/last.pt'
        if os.path.exists(last_path):
            self.__run.log_model(path=last_path, name=f'{self.args.run_name}-last-model')
        else:
            raise Exception(f'last model path is not exist at {last_path}')

        
    def get_hash(self):
        args_str = json.dumps(self.args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()

        return args_hash

    @staticmethod
    def save_json(dct, path):
        with open(path, 'w') as outfile:
            json.dump(dct, outfile)
    
    def run(self):
        not NotImplementedError()