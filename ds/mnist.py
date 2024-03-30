""" Mnist Dataset"""

from typing import List, Dict, Any
from .core import MTLDOGDS
from argparse import Namespace

import torch
from torch import Tensor
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST as mnist
from torchvision.transforms.functional import rotate

DOMAIN_TXT: List[str] = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']
DOMAIN_IDX: List[int] = [ 0,   10,   20,   30,   40,   50 ,  60,   70,   80,   90 ]
TASK_TXT: List[str] = ['rec', 'cls']


class MNIST(MTLDOGDS):
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
                ) -> MTLDOGDS:

        if root_dir is None:
            root_dir = "/".join(__file__.split("/")[:-1]) + "/src"

        super(MNIST, self).__init__(root_dir, domain, tasks, train, default_domains, default_tasks, dm2idx)

        if src_data is None:
            raise ValueError("src_data cannot be None")
        
        if src_labl is None:
            raise ValueError("src_labl cannot be None")
        
        self.data: List[Any] = src_data
        self.labl: List[Any] = src_labl

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(), 
                self.rotate_img,
                transforms.ToTensor()
            ]
        )
    
    def rotate_img(self, x) -> Tensor:
        return rotate(x, self.dm, fill=(0,), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Dict[str, Tensor]]:
        img = self.transform(self.data[index])

        tsk_dct = {}

        for tk in self.tks:
            if tk == 'rec':
                tsk_dct[tk] = img
            elif tk == 'cls':
                tsk_dct[tk] = self.labl[index]

        return img, tsk_dct

def get_mnist(args: Namespace) -> tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:

    if root_dir is None:
        root_dir = "/".join(__file__.split("/")[:-1]) + "/src"

    ori_tr = mnist(args.dt, train=True, download=True)
    ori_te = mnist(args.dt, train=False, download=True)

    ori_tr_imgs: Any | Tensor = ori_tr.data
    ori_tr_lbls: Any | Tensor = ori_tr.targets

    ori_te_imgs: Any | Tensor = ori_te.data
    ori_te_lbls: Any | Tensor = ori_te.targets

    shuffle = torch.randperm(len(ori_tr_imgs))
    ori_tr_imgs = ori_tr_imgs[shuffle]
    ori_tr_lbls = ori_tr_lbls[shuffle]

    shuffle = torch.randperm(len(ori_te_imgs))
    ori_te_imgs = ori_te_imgs[shuffle]
    ori_te_lbls = ori_te_lbls[shuffle]

    return ori_tr_imgs, ori_tr_lbls, ori_te_imgs, ori_te_lbls
    

def ds_mnist(args: Namespace) -> tuple[List[MNIST], List[MNIST]]:

    ori_tr_imgs, ori_tr_lbls, ori_te_imgs, ori_te_lbls = get_mnist(args=args)

    tr_dss = []
    te_dss = []

    for i, dmidx in enumerate(DOMAIN_IDX):
        tr_dss.append(
            MNIST(
                root_dir=args.dt, domain=dmidx, tasks=args.tkss, train=True, 
                src_data=ori_tr_imgs[i::len(DOMAIN_IDX)],
                src_labl=ori_tr_lbls[i::len(DOMAIN_IDX)]
            )
        )

        te_dss.append(
            MNIST(
                root_dir=args.dt, domain=dmidx, tasks=args.tkss, train=False, 
                src_data=ori_te_imgs[i::len(DOMAIN_IDX)],
                src_labl=ori_te_lbls[i::len(DOMAIN_IDX)]
            )
        )
    
    args.num_class = 10
    
    return args, tr_dss, te_dss