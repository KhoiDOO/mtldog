"""main.py - CLI Interaction"""

import argparse
import random, torch
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='MTLDOG - Domain Generalization for Multi-task Learning')

    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--cfp', type=int, default=0, help='Configuration path for training')

    args = parser.parse_args()

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