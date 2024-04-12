""" CityScape Dataset"""

from typing import List, Dict, Any, Tuple
from .core import MTLDOGDS
from argparse import Namespace
from torch import Tensor
from glob import glob
from torchvision import transforms
from PIL import Image

import torch.nn.functional as F
import numpy as np
import torch
import cv2
import warnings
import albumentations as A

DOMAIN_TXT: List[str] = ['normal', 'foggy', 'rainy']
DOMAIN_IDX: List[int] = [ 0,   1,   2]
TASK_TXT: List[str] = ['seg', 'depth']

def get_trans_lst():
    return [
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 10.0)),
            A.MultiplicativeNoise(),
            A.RandomRain(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.4),
            A.MedianBlur(blur_limit=5, p=0.4),
            A.Blur(blur_limit=5, p=0.4),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.12, rotate_limit=15, p=0.5,
                          border_mode = cv2.BORDER_CONSTANT),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),   
            A.Downscale(interpolation = {
                "downscale": cv2.INTER_NEAREST,
                "upscale": cv2.INTER_NEAREST
            }),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(p=0.3),
            A.ColorJitter(p=0.3),
        ], p= 0.3),
        A.RGBShift(p=0.5),
        A.RandomShadow(p=0.2)
    ]


class CityScapes(MTLDOGDS):
    def __init__(self, 
                 root_dir: str | None = None, 
                 domain: int | None = None, 
                 tasks: List[str] | None = None, 
                 train: bool = True, 
                 default_domains: List[str] = DOMAIN_TXT, 
                 default_tasks: List[str] = TASK_TXT, 
                 dm2idx: Dict[str, int] = {txt : idx for txt, idx in zip(DOMAIN_TXT, DOMAIN_IDX)},
                 size: Tuple[int] = (256, 512)) -> None:
        
        if root_dir is None:
            root_dir = "/".join(__file__.split("/")[:-1]) + "/src/cityscapes"

        super().__init__(root_dir, domain, tasks, train, default_domains, default_tasks, dm2idx)

        # self.transform = transforms.Compose([transforms.Resize((size[0], size[1])), transforms.ToTensor()])
        # self.seg_transform = transforms.Compose([transforms.Resize((size[0], size[1])), transforms.PILToTensor(), self.make_semantic_class])

        self.aug = A.Compose(get_trans_lst())

        self.semantic_map = {
            0 : ['unlabeled', 19, 'void'], 
            1 : ['ego vehicle', 19, 'void'],
            2 : ['rectification border', 19, 'void'],
            3 : ['out of roi', 19, 'void'],
            4 : ['static', 19, 'void'],
            5 : ['dynamic', 19, 'void'],
            6 : ['ground', 19, 'void'],
            7 : ['road', 0, 'flat'],
            8 : ['sidewalk', 1, 'flat'],
            9 : ['parking', 19, 'flat'],
            10 : ['rail track', 19, 'flat'],
            11 : ['building', 2, 'construction'],
            12 : ['wall', 3, 'construction'],
            13 : ['fence', 4, 'construction'],
            14 : ['guard rail', 19, 'construction'],
            15 : ['bridge', 19, 'construction'],
            16 : ['tunnel', 19, 'construction'],
            17 : ['pole', 5, 'object'],
            18 : ['polegroup', 19, 'object'],
            19 : ['traffic light', 6, 'object'],
            20 : ['traffic sign', 7, 'object'],
            21 : ['vegetation', 8, 'nature'],
            22 : ['terrain', 9, 'nature'],
            23 : ['sky', 10, 'sky'],
            24 : ['person', 11, 'human'],
            25 : ['rider', 12, 'human'],
            26 : ['car', 13, 'vehicle'],
            27 : ['truck', 14, 'vehicle'],
            28 : ['bus', 15, 'vehicle'],
            29 : ['caravan', 19, 'vehicle'],
            30 : ['trailer', 19, 'vehicle'],
            31 : ['train', 16, 'vehicle'],
            32 : ['motorcycle', 17, 'vehicle'],
            33 : ['bicycle', 18, 'vehicle'],
            34 : ['license plate', -1, 'vehicle']
        }

        subset = self.map_subset(self.tr)

        if domain == 0:
            img_dir = root_dir + f'/leftImg8bit_trainvaltest/leftImg8bit/{subset}' 
        elif domain == 1:
            img_dir = root_dir + f'/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/{subset}'
        elif domain == 2:
            img_dir = root_dir + f'/leftImg8bit_trainval_rain/leftImg8bit_rain/{subset}'
        
        self.img_paths = glob(img_dir + "/*/*")

        if len(self.img_paths) == 0:
            raise ValueError('No training images were found!')

        if domain != 0:
            data_name = np.unique(np.array(["_".join(x.split("/")[-1].split("_")[:4]) for x in self.img_paths])).tolist()

        if 'seg' in self.tks:
            self.seg_dir = self.dt + f"/gtFine_trainvaltest/gtFine/{subset}"
            self.seg_gts = glob(self.seg_dir + "/*/*_labelIds.png")
            
            if domain == 0:
                if len(self.img_paths) != len(self.seg_gts):
                    raise ValueError(f'data loss for task seg in domain {domain}, #img: {len(self.img_paths)}, while #seg ground truth: {len(self.seg_gts)}')
            elif domain == 1:
                if len(data_name) != len(self.seg_gts):
                    raise ValueError(f'data loss for task seg in domain {domain}, #img: {len(data_name)}, while #seg ground truth: {len(self.seg_gts)}')
        
        if 'depth' in self.tks:
            if domain == 2:
                self.dep_dir = self.dt + f"/leftImg8bit_trainval_rain/depth_rain/{subset}"
            else:
                self.dep_dir = self.dt + f"/disparity_trainvaltest/disparity/{subset}"
            
            self.dep_gts = glob(self.dep_dir + "/*/*")

            if domain == 0:
                if len(self.img_paths) != len(self.dep_gts):
                    raise ValueError(f'data loss for task depth in domain {domain}, #img: {len(self.img_paths)}, while #depth ground truth: {len(self.seg_gts)}')
            elif domain == 1:
                if len(data_name) != len(self.seg_gts):
                    raise ValueError(f'data loss for task depth in domain {domain}, #img: {len(data_name)}, while #seg ground truth: {len(self.seg_gts)}')
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Dict[str, Tensor]]:
        
        img_path = self.img_paths[index]
        img = self.transform(Image.open(img_path).convert("RGB"))

        filename = img_path.split("/")[-1]

        filename_split = filename.split("_")

        city_name = filename_split[0]

        img_name = "_".join(filename_split[:3])

        tsk_dct = {}

        masks = []
        names = []

        for tk in self.tks:
            if tk == 'seg':
                seg_path = self.seg_dir + f"/{city_name}/{img_name}_gtFine_labelIds.png"
                mask = Image.open(seg_path)
            elif tk == 'depth':
                if self.dm != 2:
                    depth_path = self.dep_dir + f"/{city_name}/{img_name}_disparity.png"
                else:
                    depth_path = self.dep_dir + f"/{city_name}/{img_name}_depth_rain.png"
                
                mask = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            
            masks.append(mask)
            names.append(tk)
        
        transformed = self.aug(image=img, masks=masks)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']

        
        
        return img, tsk_dct

    @staticmethod
    def map_subset(train: bool):
        return 'train' if train else 'val'
    
    def make_semantic_class(self, x):
        encx = torch.zeros(x.shape, dtype=torch.long)
        for label in self.semantic_map:
            encx[x == label] = self.semantic_map[label][1]
        onehot = F.one_hot(encx.squeeze(1), 20).permute(0, 3, 1, 2)[0].float()
        return onehot

    @staticmethod
    def process_depth(depth):
        depth[depth > 0] = (depth[depth > 0] - 1) / 256

        depth[depth == np.inf] = 0
        depth[depth == np.nan] = 0
        depth[depth < 0] = 0

        torch_depth = torch.from_numpy(depth).unsqueeze(0) / 255.0

        return torch_depth

def ds_city(args: Namespace) -> tuple[List[CityScapes], List[CityScapes]]:
    tr_dss = []
    te_dss = []

    for _, dmidx in enumerate(DOMAIN_IDX):
        tr_dss.append(
            CityScapes(root_dir=args.dt, domain=dmidx, tasks=args.tkss, train=True)
        )

        te_dss.append(
            CityScapes(root_dir=args.dt, domain=dmidx, tasks=args.tkss, train=False)
        )
    
    args.seg_num_classes = 20
    
    return args, tr_dss, te_dss

def check_args(args: Namespace, expect_domain:int):

    def UW(args: Namespace, expect_domain:int):
        raise warnings.warn("You are choosing a dataset with specific domain: {0}, but the train domains are {1}, \
                          thus automatically adjusting the args.trdms to [{2}]".format(args.ds, args.trdms, expect_domain))
    
    if len(args.trdms) > 1 or args.trdms[0] != expect_domain:
        UW(args.trdms)
    
        args.trdms = [expect_domain]

    return args

def ds_citynormal(args: Namespace) -> tuple[List[CityScapes], List[CityScapes]]:

    args = check_args(args, 0)
    
    args.seg_num_classes = 20

    return args, [CityScapes(root_dir=args.dt, domain=0, tasks=args.tkss, train=True)], [CityScapes(root_dir=args.dt, domain=0, tasks=args.tkss, train=False)]

def ds_cityfoggy(args: Namespace) -> tuple[List[CityScapes], List[CityScapes]]:

    args = check_args(args, 1)
    
    args.seg_num_classes = 20

    return args, [CityScapes(root_dir=args.dt, domain=1, tasks=args.tkss, train=True)], [CityScapes(root_dir=args.dt, domain=1, tasks=args.tkss, train=False)]

def ds_cityrainy(args: Namespace) -> tuple[List[CityScapes], List[CityScapes]]:

    args = check_args(args, 2)
    
    args.seg_num_classes = 20

    return args, [CityScapes(root_dir=args.dt, domain=2, tasks=args.tkss, train=True)], [CityScapes(root_dir=args.dt, domain=2, tasks=args.tkss, train=False)]