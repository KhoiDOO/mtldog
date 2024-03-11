""" Abstract Class for MTLDOG Dataset"""
from typing import List, Dict
import os

import torch
from torch.utils.data import Dataset


class MTLDOGDS(Dataset):
    def __init__(self, data_dir: str, domain: str, tasks: List[str], default_domains: List[str], default_tasks: List[str]) -> None:
        super(MTLDOGDS, self).__init__()
        self.dt = data_dir
        self.dm = domain
        self.tks = tasks
        self.def_dms = default_domains
        self.def_tks = default_tasks

        self.check_dm()
        self.check_tk()
    
    @staticmethod
    def check_exists(path: str) -> bool:
        return os.path.exists(path)

    def check_dt(self):
        if not self.check_exists(self.dt):
            raise ValueError(f"the input data directory {self.dt} is not existed")

    def check_dm(self):
        if self.dm not in self.def_dms:
            raise ValueError(f"domain {self.dm} is not available in {self.def_dms}")
    
    def check_tk(self):
        if len(self.tks) > len(self.def_dms):
            raise ValueError(f"the number of input tasks ({len(self.tks)}) is more than available domains ({len(self.def_tks)})")

        for tk in self.tks:
            if tk not in self.def_dms:
                raise ValueError(f"domain {tk} is not available in {self.def_dms}")
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return super().__getitem__(index)

    def __len__(self):
        raise NotImplementedError()