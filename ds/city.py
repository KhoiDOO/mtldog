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

        if domain == 0:
            if self.tr:
                img_dir = root_dir + '/leftImg8bit_trainvaltest/leftImg8bit/train'
            else:
                img_dir = root_dir + '/leftImg8bit_trainvaltest/leftImg8bit/val'
        elif domain == 1:
            if self.tr:
                img_dir = root_dir + '/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/train'
            else:
                img_dir = root_dir + '/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/val'
        elif domain == 2:
            if self.tr:
                img_dir = root_dir + '/leftImg8bit_trainval_rain/leftImg8bit_rain/train'
            else:
                img_dir = root_dir + '/leftImg8bit_trainval_rain/leftImg8bit_rain/val'
        
        self.img_paths = glob(img_dir + "/*/*")
    
    def __len__(self):
        return len(self.img_paths)

def ds_city(args: Namespace) -> tuple[List[CityScapes], List[CityScapes]]:
    pass

# CHECK

"""
ROOT = "/media/mountHDD3/data_storage/cityscapes/unzip"

tr_norm_ds = CityScapes(root_dir=ROOT, domain=0, train=True)
tr_fogg_ds = CityScapes(root_dir=ROOT, domain=1, train=True)
tr_rain_ds = CityScapes(root_dir=ROOT, domain=2, train=True)

te_norm_ds = CityScapes(root_dir=ROOT, domain=0, train=False)
te_fogg_ds = CityScapes(root_dir=ROOT, domain=1, train=False)
te_rain_ds = CityScapes(root_dir=ROOT, domain=2, train=False)

print(len(tr_norm_ds))
print(len(tr_fogg_ds))
print(len(tr_rain_ds))

print(len(te_norm_ds))
print(len(te_fogg_ds))
print(len(te_rain_ds))
"""