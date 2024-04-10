from argparse import Namespace

import os
import argparse
import random, torch
import numpy as np
import json

def prepare():
    cache_dir = os.environ['WANDB_CACHE_DIR'] = os.getcwd() + "/.cache"
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    wandb_cache_dir = cache_dir + "/wandb"
    if not os.path.exists(wandb_cache_dir):
        os.mkdir(wandb_cache_dir)
    
    os.environ['WANDB_CACHE_DIR'] = wandb_cache_dir

def train_init(args):
    if args.tm == 'sup':
        from train import SUP
        trainer = SUP(args=args)
    elif args.tm == 'sinsup':
        from train import SINSUP
        trainer = SINSUP(args=args)
    else:
        raise ValueError(f"training mode {args.tm} is not available")

    trainer.run()

if __name__ == "__main__":

    prepare()

    parser = argparse.ArgumentParser(prog='MTLDOG - Domain Generalization for Multi-task Learning')

    parser.add_argument('--cfp', type=str, default="./hparams/example.json", help='Configuration path for training')

    args = parser.parse_args()

    cfdct = json.load(open(args.cfp, 'r'))
    args = Namespace(**cfdct)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_init(args=args)