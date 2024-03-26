""" CityScape Dataset"""

from typing import List, Dict, Any
from .core import MTLDOGDS
from argparse import Namespace
from torch import Tensor
from glob import glob

import torch

DOMAIN_TXT: List[str] = ['normal', 'foggy', 'rainy']
DOMAIN_IDX: List[int] = [ 0,   1,   2]
TASK_TXT: List[str] = ['seg', 'depth']


class CityScapes(MTLDOGDS):
    def __init__(self, 
                 root_dir: str | None = None, 
                 domain: int | None = None, 
                 tasks: List[str] | None = None, 
                 train: bool = True, 
                 default_domains: List[str] = DOMAIN_TXT, 
                 default_tasks: List[str] = TASK_TXT, 
                 dm2idx: Dict[str, int] = {txt : idx for txt, idx in zip(DOMAIN_TXT, DOMAIN_IDX)}) -> None:
        
        if root_dir is None:
            root_dir = "/".join(__file__.split("/")[:-1]) + "/src/cityscapes"

        super().__init__(root_dir, domain, tasks, train, default_domains, default_tasks, dm2idx)

        if domain == 'normal':
            if self.tr:
                img_dir = root_dir + '/leftImg8bit_trainvaltest/leftImg8bit/train'
            else:
                img_dir = root_dir + '/leftImg8bit_trainvaltest/leftImg8bit/val'
        elif domain == 'foggy':
            if self.tr:
                img_dir = root_dir + '/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/train'
            else:
                img_dir = root_dir + '/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/val'
        elif domain == 'rainy':
            if self.tr:
                img_dir = root_dir + '/leftImg8bit_trainval_rain/leftImg8bit_rain/train'
            else:
                img_dir = root_dir + '/leftImg8bit_trainval_rain/leftImg8bit_rain/val'
        
        img_paths = glob(img_dir + "/*/*")

