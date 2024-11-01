""" Abstract Class for MTLDOG Dataset"""
from typing import List, Dict
import os

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MTLDOGDS(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 domain: int, 
                 tasks: List[str], 
                 train: bool,
                 default_domains: List[str], 
                 default_tasks: List[str],
                 dm2idx: Dict[str, int]
                ) -> None:
        super(MTLDOGDS, self).__init__()
        self.dt = root_dir
        self.dm = domain
        self.tks = tasks
        self.tr = train
        self.def_dms = default_domains
        self.def_tks = default_tasks
        self.dm2idx = dm2idx
        self.idx2dm = {self.dm2idx[key] : key for key in self.dm2idx}
        self.dmtxt = self.idx2dm[self.dm]

        self.check_dm()
        self.check_tk()
    
    @staticmethod
    def check_exists(path: str) -> bool:
        return os.path.exists(path)

    def check_dt(self):
        if not self.check_exists(self.dt):
            raise ValueError(f"the input data directory {self.dt} is not existed")

    def check_dm(self):
        if self.dm is None:
            raise ValueError('domain cannot be None')
        if self.dm not in self.idx2dm:
            raise ValueError(f"domain {self.dm} is not available in {self.def_dms}")
    
    def check_tk(self):
        if self.tks is None:
            raise ValueError(f"tasks cannot be None")
        if len(self.tks) > len(self.def_tks):
            raise ValueError(f"the number of input tasks ({len(self.tks)}) is more than available tasks ({len(self.def_tks)}), \
                             the available tasks are: {self.def_tks}")

        for tk in self.tks:
            if tk not in self.def_tks:
                raise ValueError(f"task {tk} is not available in {self.def_tks}")
    
    def __getitem__(self, index: int) -> tuple[Tensor, Dict[str, Tensor]]:
        return super().__getitem__(index)

    def __len__(self):
        raise NotImplementedError()

    @property
    def domain_idx(self) -> int:
        return self.dm
    
    @property
    def domain_txt(self) -> str:
        return self.dmtxt
    
    # @property
    # def dm2idx(self) -> Dict[str, int]:
    #     return self.dm2idx

    # @property
    # def idx2dm(self) -> Dict[int, str]:
    #     return self.idx2dm

    @property
    def tasks(self) -> List[str]:
        return self.tks

    @property
    def train(self) -> bool:
        return self.train