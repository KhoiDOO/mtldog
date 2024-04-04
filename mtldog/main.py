from argparse import Namespace

import argparse
import random, torch
import numpy as np
import json


if __name__ == "__main__":
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

    if args.tm == 'sup':
        from train import SUP
        trainer = SUP(args=args)
    else:
        raise ValueError(f"training mode {args.tm} is not available")

    trainer.run()