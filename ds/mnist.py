""" Mnist Dataset"""

from typing import List, Dict, Any
from .core import MTLDOGDS
from argparse import Namespace

import torch
from torch import Tensor
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms.functional import rotate

DOMAIN_TXT = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']
DOMAIN_IDX = [ 0,   10,   20,   30,   40,   50 ,  60,   70,   80,   90 ]
TASK_TXT = ['rec', 'cls']

class MTLRotMnist(MTLDOGDS):
    def __init__(self, 
                 root_dir: str | None = None, 
                 domain: int | None = None, 
                 tasks: List[str] | None = None, 
                 train: bool = True,
                 default_domains: List[str] = DOMAIN_TXT,
                 default_tasks: List[str] = TASK_TXT,
                 dm2idx: Dict[str, int] = {txt : idx for txt, idx in zip(DOMAIN_TXT, DOMAIN_IDX)},
                 src_data: List[Any] = None,
                 src_labl: List[Any] = None,
                ) -> None:

        if root_dir is None:
            root_dir = "/".join(__file__.split("/")[:-1]) + "/source"

        super(MTLRotMnist, self).__init__(root_dir, domain, tasks, train, default_domains, default_tasks, dm2idx)

        if src_data is None:
            raise ValueError("src_data cannot be None")
        
        if src_labl is None:
            raise ValueError("src_labl cannot be None")
        
        self.data = src_data
        self.labl = src_labl

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: rotate(x, self.dm, fill=(0,), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
                transforms.ToTensor()
            ]
        )
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, index: int) -> tuple[Tensor, Dict[str, Tensor]]:
        img = self.transform(self.data[index])

        return img, {"rec" : img, "cls" : self.labl[index]}
    

def ds_mtlrotmnist(args: Namespace) -> tuple[List[Tensor], List[Tensor]]:

    if args.dt is None:
        args.dt = "/".join(__file__.split("/")[:-1]) + "/source"

    ori_tr = MNIST(args.dt, train=True, download=True)
    ori_te = MNIST(args.dt, train=False, download=True)

    ori_tr_imgs = ori_tr.data
    ori_tr_lbls = ori_tr.targets

    ori_te_imgs = ori_te.data
    ori_te_lbls = ori_te.targets

    shuffle = torch.randperm(len(ori_tr_imgs))
    ori_tr_imgs = ori_tr_imgs[shuffle]
    ori_tr_lbls = ori_tr_lbls[shuffle]

    shuffle = torch.randperm(len(ori_te_imgs))
    ori_te_imgs = ori_te_imgs[shuffle]
    ori_te_lbls = ori_te_lbls[shuffle]

    tr_dss = []
    te_dss = []

    for i, dmidx in enumerate(DOMAIN_IDX):
        tr_dss.append(
            MTLRotMnist(
                root_dir=args.dt, domain=dmidx, tasks=args.trtks, train=True, 
                src_data=ori_tr_imgs[i::len(DOMAIN_IDX)],
                src_labl=ori_tr_lbls[i::len(DOMAIN_IDX)]
            )
        )

        te_dss.append(
            MTLRotMnist(
                root_dir=args.dt, domain=dmidx, tasks=args.trtks, train=True, 
                src_data=ori_te_imgs[i::len(DOMAIN_IDX)],
                src_labl=ori_te_lbls[i::len(DOMAIN_IDX)]
            )
        )
    
    return tr_dss, te_dss