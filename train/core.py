from typing import *
from argparse import Namespace
from statistics import mean
from torch import Tensor

import os, sys, torch, wandb
import json, hashlib
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import ds, arch, algo, loss, metric
import torch.distributed as dist


class MTLDOGTR:
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.prepare_device()
        self.prepare_ds()
        self.prepare_save_dir()
        self.prepare_algo()
        if self.args.wandb:
            self.prepare_wandb()
        self.prepare_log()
        self.prepare_loss()
        self.prepare_metric()

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

        self.args, self.tr_dss, self.te_dss = self.ds_dct[f'ds_{self.args.ds}'](self.args)

        assert self.args.bs % self.args.world_size == 0

        self.args.task_num = len(self.args.tkss)

    def prepare_save_dir(self):
        main_dir = os.getcwd()

        run_dir = main_dir + '/runs'
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        
        self.method_dir = run_dir + f'/{self.args.m}'
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
    
    def prepare_algo(self):
        algo_map = vars(algo)
        self.algo_dct = {k : algo_map[k] for k in algo_map if 'algo' in k}
        self.algo = self.algo_dct[f'algo_{self.args.m}']()

        model_map = vars(arch)
        self.model_dct = {k : model_map[k] for  k in model_map if 'model' in k}
        self.model = self.model_dct[f'model_{self.args.model}']()

    def prepare_wandb(self):
        self.args.run_name = f'{self.args.m}__{self.args.ds}__{self.args.hashcode}'

        hparams_path = self.args.hp
        params = json.load(open(hparams_path, 'r'))
        config_dict = vars(self.args)
        config_dict.update(params)

        self.logrun = wandb.init(
            project=self.args.wandb_prj,
            entity=self.args.wandb_entity,
            group="DDP",
            config=config_dict,
            name=self.args.run_name,
            force=True
        )

    def prepare_log(self):
        self.log_dict = {}
        self.log_grad_dict = {}
    
    def prepare_loss(self):
        loss_map = vars(loss)
        self.loss_dct = {k : loss_map[k] for k in loss_map if 'loss' in k and k.split('_')[-1] in self.args.losses and k.split('_')[-2] in self.args.tkss}
    
    def prepare_metric(self):
        metric_map = vars(metric)
        self.metric_dct = {k : metric_map[k] for k in metric_map if 'metric' in k and k.split('_')[-2] in self.args.tkss}
    
    def log_wbmodel(self):
        best_path = self.args.save_dir + f'/best.pt'
        if os.path.exists(best_path):
            self.logrun.log_model(path=best_path, name=f'{self.args.run_name}-best-model')
        else:
            raise Exception(f'best model path is not exist at {best_path}')
        
        last_path = self.args.save_dir + f'/last.pt'
        if os.path.exists(last_path):
            self.logrun.log_model(path=last_path, name=f'{self.args.run_name}-last-model')
        else:
            raise Exception(f'last model path is not exist at {last_path}')

    def get_hash(self):
        args_str = json.dumps(vars(self.args), sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        return args_hash
    
    def track(self, key: str, value: float):
        if key in self.log_dict:
            self.log_dict[key].append(value)
        else:
            self.log_dict[key] = [value]
        
    def sync(self):
        mean_log_dict = {k : mean(self.log_dict[k]) for k in self.log_dict}
        self.logrun.log(mean_log_dict)
        self.log_dict = {}
    
    def track_grad(self, key: str, value: Dict[str, Tensor]):
        self.log_grad_dict[key] = value

    def sync_grad():
        pass
    
    def show_log_dict(self, epoch: int, stage:str):
        print(f"EPOCH: {epoch} ~~~ {stage}")
        print("{:<40} {:<40}".format('KEY', 'VALUE'))
        mean_log_dict = {k : mean(self.log_dict[k]) for k in self.log_dict}
        for key, value in mean_log_dict.items():
            print("{:<40} {:<40}".format(key, value))
        self.log_dict = {}

    @staticmethod
    def save_json(dct, path):
        with open(path, 'w') as outfile:
            json.dump(dct, outfile)
    
    def run(self):
        not NotImplementedError()
    
    @staticmethod
    def cleanup():
        dist.destroy_process_group()