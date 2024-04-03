from typing import *
from argparse import Namespace
from statistics import mean
from torch import Tensor
from .utils import *
from pandas import DataFrame
from glob import glob
from alive_progress import alive_it
from fastparquet import write

import os, sys, torch, wandb
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
        self.log = {}
        self.round = 0
        if self.args.wandb:
            self.prepare_wandb()
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

        self.args.hashcode = self.hash = get_hash(args=self.args)
        self.args.save_dir = self.save_dir = run_dir + f"/{self.args.hashcode}"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.args.best_model_path = self.best_model_path = self.save_dir + f'/best.pt'
        self.args.last_model_path = self.last_model_path = self.save_dir + f'/last.pt'
    
    def prepare_algo(self):
        algo_map = vars(algo)
        self.algo_dct = {k : algo_map[k] for k in algo_map if 'algo' in k}
        self.algo = self.algo_dct[f'algo_{self.args.m}']()

        model_map = vars(arch)
        self.model_dct = {k : model_map[k] for  k in model_map if 'model' in k}
        self.model = self.model_dct[f'model_{self.args.model}']()

    def prepare_wandb(self):
        self.args.run_name = self.run_name = f'{self.args.hashcode}'

        self.logrun = wandb.init(
            project=self.args.wandb_prj,
            entity=self.args.wandb_entity,
            config=self.args,
            name=self.args.run_name,
            force=True
        )

        self.logart = []
    
    def prepare_loss(self):
        loss_map = vars(loss)
        self.loss_dct = {k : loss_map[k] for k in loss_map if 'loss' in k and k.split('_')[-1] in self.args.losses and k.split('_')[-2] in self.args.tkss}
    
    def prepare_metric(self):
        metric_map = vars(metric)
        self.metric_dct = {k : metric_map[k] for k in metric_map if 'metric' in k and k.split('_')[-2] in self.args.tkss}
    
    def prepare_grad_logging(self):
        if self.args.grad:
            if len(self.args.tkss) == 1 and len(self.args.trdms) == 1:
                self.args.grad = False
                print("Force <args.grad> to be False as len(args.tkss) = 1 and len(rgs.trdms) = 1")
    
    def track(self, key: str, value: float):
        if key in self.log:
            self.log[key].append(value)
        else:
            self.log[key] = [value]
    
    def log_extract(self, grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]],
                sol_grad_share: Tensor, sol_grad_head: Dict[str, Tensor], 
                hess_dict: Dict[str, Dict[str, Dict[str, Dict[str, List[Tensor]]]]]):
        
        mean_log = {k : mean(self.log[k]) for k in self.log}
        
        main_grad_dict = preprocess_grad_train(grad_dict=grad_dict, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head, args=self.args)

        grad_hess_dict = preprocess_grad_hess_adv(hess_dict=hess_dict, args=self.args)
        
        mean_log.update(main_grad_dict)

        mean_log.update(grad_hess_dict)
        
        self.log = {}

        return mean_log

    def postprocess_log(self, log_dict: Dict[str, Tensor | DataFrame| float]):

        copy_log_dict = log_dict.copy()

        for key in copy_log_dict:
            if 'tab' in key:
                copy_log_dict[key] = wandb.Table(dataframe=copy_log_dict[key])

        non_vec_dict = {key: value for key, value in copy_log_dict.items() if 'vec' not in key}

        return non_vec_dict

    
    def sync(self, grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]] | None = None,
                sol_grad_share: Tensor | None = None, 
                sol_grad_head: Dict[str, Tensor] | None = None, 
                hess_dict: Dict[str, Dict[str, Dict[str, Dict[str, List[Tensor]]]]] | None = None):
        
        mean_log = self.log_extract(grad_dict=grad_dict, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head, hess_dict=hess_dict)
        nonvec_dict = self.postprocess_log(mean_log)

        raw_path = self.save_dir + f'/main_log_round_{self.round}.xz'
        save_pickle(dct=mean_log, path=raw_path)
        
        if self.args.wandb:
            self.logart.append(raw_path)
            if not self.args.synclast:
                self.logrun.log(nonvec_dict)
            else:
                nonvec_path = self.save_dir + f'/nonvec_log_round_{self.round}.xz'
                save_pickle(dct=nonvec_dict, path=nonvec_path)
        elif self.args.verbose:
            show_log(mean_log, self.round, self.args)
        
        self.round += 1
    
    def sync_finish(self):
        logart = wandb.Artifact(name=f"raw_log_{self.hash}", type="log")
        for raw_path in self.logart:
            logart.add_file(local_path=raw_path)
        self.logrun.log_artifact(logart)

        if self.args.synclast:
            nonvec_file_paths = glob(self.save_dir + "/nonvec_*")

            for nonvec_path in nonvec_file_paths:
                nonvec_dict = read_pickle(nonvec_path)
                self.logrun.log(nonvec_dict)
        
        if os.path.exists(self.best_model_path):
            self.logrun.log_model(path=self.best_model_path, name=f'{self.run_name}-best-model')
        else:
            raise Exception(f'best model path is not exist at {self.best_model_path}')
        
        if os.path.exists(self.last_model_path):
            self.logrun.log_model(path=self.last_model_path, name=f'{self.run_name}-last-model')
        else:
            raise Exception(f'last model path is not exist at {self.last_model_path}')

    
    def run(self):
        not NotImplementedError()
    
    @staticmethod
    def finish():
        dist.destroy_process_group()