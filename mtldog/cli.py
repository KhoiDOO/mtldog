from main import prepare, train_init

import argparse
import random, torch
import numpy as np


if __name__ == "__main__":

    prepare()

    parser = argparse.ArgumentParser(prog='MTLDOG - Domain Generalization for Multi-task Learning')

    # dataset
    parser.add_argument('--ds', type=str, required=True, help="dataset")
    parser.add_argument('--dt', type=str, help='data dir')
    parser.add_argument('--bs', type=int, required=True, help='batch size')
    parser.add_argument('--wk', type=str, default=8, help='number of workers')
    parser.add_argument('--pm', action='store_true', help='pin memory')
    parser.add_argument('--sz', type=int, nargs='+', required=False, help='size of processed image (h, w)')

    # domain - tasks - losses
    parser.add_argument('--trdms', type=int, nargs='+', required=True, default=[0], help='train domains')
    parser.add_argument('--tkss', type=str, nargs='+', required=True, help='training tasks')
    parser.add_argument('--losses', type=str, nargs='+', required=True, help='training losses')
    
    # method
    parser.add_argument('--m', type=str, required=True, help='method')
    parser.add_argument('--model', type=str, required=True, help='model')
    parser.add_argument('--at', type=str, required=True, help='archiecture')
    parser.add_argument('--bb', type=str, required=True, help='backbone')

    # training
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--tm', type=str, default='sup', help='training mode')
    parser.add_argument('--dvids', type=str, nargs='+', default=[0], help='devices')
    parser.add_argument('--round', type=int, required=True, help='number of rounds')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--port', type=int, default=8080, help='nccl port')
    parser.add_argument('--chkfreq', type=int, required=True, help='checkpoint frequency')

    # performance
    parser.add_argument('--mehigh', action='store_true', help='higher metric value is better')
    parser.add_argument('--mecp', type=str, required=True, help='checkpoint metric')

    # logging
    parser.add_argument('--wandb', action='store_true', help='toggle to use wandb')
    parser.add_argument('--wandb_prj', type=str, required=False, help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, required=False, help='wandb entity name')

    # analysis
    parser.add_argument('--analysis', action='store_true', help='Enable analysis')
    parser.add_argument('--quant', action='store_true', help='Disable quant')
    parser.add_argument('--spcnt', type=int, default=1, help='Set spcnt value')
    parser.add_argument('--diseval', action='store_true', help='Disable diseval')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose')
    parser.add_argument('--extend_verbose', action='store_true', help='Enable extend_verbose')
    parser.add_argument('--synclast', action='store_true', help='Enable synclast')

    # losses
    parser.add_argument('--gamma', type=float, required=False, help='gamma for Focal')
    parser.add_argument('--blvs', type=float, required=False, help='s param for BLV')
    parser.add_argument('--tau', type=float, required=False, help='temp param for gumbel')
    parser.add_argument('--ldamm', type=float, required=False, help='m param for ldam')
    parser.add_argument('--ldams', type=float, required=False, help='s param for ldam')

    # methods
    parser.add_argument('--cagradc', type=float, required=False, help='c param for cagrad')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_init(args=args)