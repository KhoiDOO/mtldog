from typing import *
from argparse import Namespace
from statistics import mean
from torch import Tensor
from .utils import get_hash, save_json, cosine_dataframe, dot_dataframe
from pandas import DataFrame
from tabulate import tabulate

import os, sys, torch, wandb, json
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
        
        self.method_dir = run_dir + f'/{self.args.m}'
        if not os.path.exists(self.method_dir):
            os.mkdir(self.method_dir)
        
        self.ds_dir = self.method_dir + f'/{self.args.ds}'
        if not os.path.exists(self.ds_dir):
            os.mkdir(self.ds_dir)

        self.args.hashcode = get_hash(args=self.args)
        self.args.save_dir = self.save_dir = self.ds_dir + f"/{self.args.hashcode}"
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
        self.args.run_name = self.run_name = f'{self.args.m}__{self.args.ds}__{self.args.hashcode}'

        hparams_path = self.args.hp
        params = json.load(open(hparams_path, 'r'))
        self.config_dict = vars(self.args)
        self.config_dict.update(params)

        self.logrun = wandb.init(
            project=self.args.wandb_prj,
            entity=self.args.wandb_entity,
            config=self.config_dict,
            name=self.args.run_name,
            force=True
        )
    
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
    
    def save_model(self):
        pass
    
    def log_wbmodel(self):
        if os.path.exists(self.best_model_path):
            self.logrun.log_model(path=self.best_model_path, name=f'{self.run_name}-best-model')
        else:
            raise Exception(f'best model path is not exist at {self.best_model_path}')
        
        if os.path.exists(self.last_model_path):
            self.logrun.log_model(path=self.last_model_path, name=f'{self.run_name}-last-model')
        else:
            raise Exception(f'last model path is not exist at {self.last_model_path}')
    
    def track(self, key: str, value: float):
        if key in self.log:
            self.log[key].append(value)
        else:
            self.log[key] = [value]
    
    def log_extract(self, grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]],
                sol_grad_share: Tensor, sol_grad_head: Dict[str, Tensor]):
        
        mean_log = {k : mean(self.log[k]) for k in self.log}
        
        main_grad_dict = self.preprocess_grad_train(grad_dict=grad_dict, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head)
        
        mean_log.update(main_grad_dict)
        
        self.log = {}
        return mean_log

    
    def sync(self, grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]],
                sol_grad_share: Tensor, sol_grad_head: Dict[str, Tensor]):

        mean_log = self.log_extract(grad_dict=grad_dict, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head)

        for key in mean_log:
            if 'mat' in key:
                mean_log[key] = wandb.Table(dataframe=mean_log[key])

        self.logrun.log(mean_log)
    
    def logging(self, grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]],
                sol_grad_share: Tensor, sol_grad_head: Dict[str, Tensor], round:int):
        
        mean_log = self.log_extract(grad_dict=grad_dict, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head)

        print(f"ROUND: {round}")
        print("{:<70} {:<70}".format('KEY', 'VALUE'))
        print("*"*140)
        for key, value in mean_log.items():
            if not isinstance(value, Tensor) and not (isinstance(value, DataFrame)):
                print("{:<70} {:<70}".format(key, value))
                print("-"*140)
        
        for key, value in mean_log.items():
            if isinstance(value, DataFrame):
                print(key)
                print(tabulate(value, headers='keys', tablefmt='psql'))
    
    def preprocess_grad_train(self, 
                              grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]], 
                              sol_grad_share: Tensor, 
                              sol_grad_head: Dict[str, Tensor]) -> Dict[str, Tensor | DataFrame]:
        
        main_grad_dict = {}
        if grad_dict is not None:
            share_grad_dict = {dmtxt : grad_dict[dmtxt]['share'] for dmtxt in grad_dict}
            head_grad_dict = {dmtxt : grad_dict[dmtxt]['heads'] for dmtxt in grad_dict}

            share_grad_keys = []
            share_grad_vectors = []
            for dmtxt in share_grad_dict:
                for tkidx, tk in enumerate(self.args.tkss):
                    share_grad_keys.append(f'{dmtxt}-{tk}')
                    share_grad_vectors.append(share_grad_dict[dmtxt][tkidx])

                    main_grad_dict[f'grad-share-{dmtxt}-{tk}-vec'] = share_grad_dict[dmtxt][tkidx]
            
            main_grad_dict['grad-share-cos-mat'] = cosine_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)
            main_grad_dict['grad-share-dot-mat'] = dot_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)

            if sol_grad_share is not None:
                main_grad_dict[f'sol-grad-share-vec'] = sol_grad_share
                share_grad_keys.append('sol-grad')
                share_grad_vectors.append(sol_grad_share)

                main_grad_dict['sol-grad-share-cos-mat'] = cosine_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)
                main_grad_dict['sol-grad-share-dot-mat'] = dot_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)

            for _, tk in enumerate(self.args.tkss):
                head_grad_keys = []
                head_grad_vectors = []
                for dmtxt in head_grad_dict:
                    head_grad_keys.append(f'{dmtxt}-{tk}')
                    head_grad_vectors.append(head_grad_dict[dmtxt][tk])
                
                    main_grad_dict[f'grad-heads-{dmtxt}-{tk}-vec'] = head_grad_dict[dmtxt][tk]

                main_grad_dict['grad-heads-cos-mat'] = cosine_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
                main_grad_dict['grad-heads-dot-mat'] = dot_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)

                if sol_grad_head is not None:
                    main_grad_dict[f'sol-grad-head-{tk}-vec'] = sol_grad_head[tk]
                    head_grad_keys.append(f'sol-grad-head-{tk}-vec')
                    head_grad_vectors.append(sol_grad_head[tk])

                    main_grad_dict[f'sol-grad-heads-{tk}-cos-mat'] = cosine_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
                    main_grad_dict[f'sol-grad-heads-{tk}-dot-mat'] = dot_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
        
        return main_grad_dict

    def track_sync_grad_train(self, 
                              grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]], 
                              sol_grad_share: Tensor, 
                              sol_grad_head: Dict[str, Tensor]):
        
        main_grad_dict = self.preprocess_grad_train(grad_dict=grad_dict, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head)

        for key, item in main_grad_dict.items():
            if 'vec' in key:
                self.logrun.log({key : item})
            elif 'mat' in key:
                self.logrun.log({key : wandb.Table(dataframe=item)})
            else:
                print(key)

    def show_table_grad_train(self, 
                              grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]], 
                              sol_grad_share: Tensor, 
                              sol_grad_head: Dict[str, Tensor]):
        
        main_grad_dict = self.preprocess_grad_train(grad_dict=grad_dict, sol_grad_share=sol_grad_share, sol_grad_head=sol_grad_head)

        mat_main_grad_dict = {key : item for key, item in main_grad_dict.items() if 'mat' in key}

        for key, item in mat_main_grad_dict.items():
            print(key)
            print(tabulate(item, headers='keys', tablefmt='psql'))
    
    def show_log(self, round: int, stage:str):
        print(f"ROUND: {round} ~~~ {stage}")
        print("{:<40} {:<40}".format('KEY', 'VALUE'))
        mean_log = {k : mean(self.log[k]) for k in self.log}
        for key, value in mean_log.items():
            print("{:<40} {:<40}".format(key, value))
        self.log = {}
    
    def run(self):
        not NotImplementedError()
    
    @staticmethod
    def finish():
        dist.destroy_process_group()