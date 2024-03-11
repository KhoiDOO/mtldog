""" Mnist Dataset"""

from typing import List, Dict, Any
from .core import MTLDOGDS

import torch
from torchvision.datasets import MNIST


class MTLRotMnist(MTLDOGDS):
    def __init__(self, 
                 data_dir: str = None, 
                 domain: List[str] = None, 
                 tasks: List[str] = None, 
                 default_domains: List[str] = [0, 15, 30, 45, 60, 75], 
                 default_tasks: List[str] = ['rec', 'cls'],
                 src_data: List[Any] = None,
                 src_labl: List[Any] = None,
                ) -> None:

        if data_dir is None:
            data_dir = "/".join(__file__.split("/")[:-1]) + "/source"

        super().__init__(data_dir, domain, tasks, default_domains, default_tasks)

        if src_data is None:
            raise ValueError("src_data cannot be None")
        if src_labl is None:
            raise ValueError("src_labl cannot be None")
        
        self.src_data = src_data
        self.src_labl = src_labl
