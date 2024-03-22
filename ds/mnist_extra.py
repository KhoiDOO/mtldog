from typing import Any, Dict, List
from .core import MTLDOGDS
from .mnist import DOMAIN_IDX, DOMAIN_TXT, TASK_TXT
from .mnist import *

MNISTEASY_DOMAIN_TXT: List[str] = DOMAIN_TXT[:5]
MNISTEASY_DOMAIN_IDX: List[int] = DOMAIN_IDX[:5]

class MNISTEASY(MNIST):
    def __init__(self, root_dir: str | None = None, domain: int | None = None, tasks: List[str] | None = None, train: bool = True, 
                 default_domains: List[str] = MNISTEASY_DOMAIN_TXT, 
                 default_tasks: List[str] = TASK_TXT, 
                 dm2idx: Dict[str, int] = {txt : idx for txt, idx in zip(MNISTEASY_DOMAIN_TXT, MNISTEASY_DOMAIN_IDX)}, 
                 src_data: List[Any] = None, 
                 src_labl: List[Any] = None) -> MTLDOGDS:
        super().__init__(root_dir, domain, tasks, train, default_domains, default_tasks, dm2idx, src_data, src_labl)

MNISTMED_DOMAIN_TXT: List[str] = [str(x) for x in range(0, 60, 6)]
MNISTMED_DOMAIN_IDX: List[int] = [x for x in range(0, 60, 6)]

class MNISTMED(MNIST):
    def __init__(self, root_dir: str | None = None, domain: int | None = None, tasks: List[str] | None = None, train: bool = True, 
                 default_domains: List[str] = MNISTMED_DOMAIN_TXT, 
                 default_tasks: List[str] = TASK_TXT, 
                 dm2idx: Dict[str, int] = {txt : idx for txt, idx in zip(MNISTMED_DOMAIN_TXT, MNISTMED_DOMAIN_IDX)}, 
                 src_data: List[Any] = None, 
                 src_labl: List[Any] = None) -> MTLDOGDS:
        super().__init__(root_dir, domain, tasks, train, default_domains, default_tasks, dm2idx, src_data, src_labl)


def ds_mnisteasy(args: Namespace):

    ori_tr_imgs, ori_tr_lbls, ori_te_imgs, ori_te_lbls = get_mnist(args=args)

    tr_dss = []
    te_dss = []

    for i, dmidx in enumerate(MNISTEASY_DOMAIN_IDX):
        tr_dss.append(
            MNISTEASY(
                root_dir=args.dt, domain=dmidx, tasks=args.tkss, train=True, 
                src_data=ori_tr_imgs[i::len(MNISTEASY_DOMAIN_IDX)],
                src_labl=ori_tr_lbls[i::len(MNISTEASY_DOMAIN_IDX)]
            )
        )

        te_dss.append(
            MNISTEASY(
                root_dir=args.dt, domain=dmidx, tasks=args.tkss, train=False, 
                src_data=ori_te_imgs[i::len(MNISTEASY_DOMAIN_IDX)],
                src_labl=ori_te_lbls[i::len(MNISTEASY_DOMAIN_IDX)]
            )
        )
    
    args.num_class = 10
    
    return args, tr_dss, te_dss

def ds_mnistmed(args: Namespace):

    ori_tr_imgs, ori_tr_lbls, ori_te_imgs, ori_te_lbls = get_mnist(args=args)

    tr_dss = []
    te_dss = []

    for i, dmidx in enumerate(MNISTMED_DOMAIN_IDX):
        tr_dss.append(
            MNISTMED(
                root_dir=args.dt, domain=dmidx, tasks=args.tkss, train=True, 
                src_data=ori_tr_imgs[i::len(MNISTMED_DOMAIN_IDX)],
                src_labl=ori_tr_lbls[i::len(MNISTMED_DOMAIN_IDX)]
            )
        )

        te_dss.append(
            MNISTMED(
                root_dir=args.dt, domain=dmidx, tasks=args.tkss, train=False, 
                src_data=ori_te_imgs[i::len(MNISTMED_DOMAIN_IDX)],
                src_labl=ori_te_lbls[i::len(MNISTMED_DOMAIN_IDX)]
            )
        )
    
    args.num_class = 10
    
    return args, tr_dss, te_dss