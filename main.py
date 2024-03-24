"""main.py - CLI Interaction"""

import argparse
import random, torch
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='MTLDOG - Domain Generalization for Multi-task Learning')

    # dataset
    parser.add_argument('--ds', type=str, required=True, help='dataset in use')
    parser.add_argument('--dt', type=str, required=False, help='root data directory')
    parser.add_argument('--bs', type=int, required=True, default=64, help='batch size')
    parser.add_argument('--wk', type=int, required=True, help='no. ds worker')
    parser.add_argument('--pm', action='store_true', help='toggle to use pin memory')

    # domain - tasks - losses
    parser.add_argument('--trdms', type=int, nargs='+', required=True, default=[0], help='list of domain used in training')
    parser.add_argument('--tkss', type=str, nargs='+', required=True, help='list of tasks used in training')
    parser.add_argument('--losses', type=str, nargs='+', required=True, help='losses of tasks used in training')

    # method
    parser.add_argument('--m', type=str, required=True, help='method used in training')
    parser.add_argument('--hp', type=str, required=True, help='json file path for hyper-parameter of method')

    # model
    parser.add_argument('--model', type=str, required=True, help='model type (i.e. ae, hps (hard parameter sharing))')
    parser.add_argument('--at', type=str, required=True, help='archiecture type (i.e. ae, unet)')
    parser.add_argument('--bb', type=str, required=True, help='backbone type (i.e. ae, base, resnet18)')

    # training
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--tm', type=str, default='sup', help='training mode (i.e. sup (supervised))')
    parser.add_argument('--dvids', type=str, nargs='+', default=[0], help='list of device used in training')
    parser.add_argument('--round', type=int, default=1, help='number of rounds used in training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--port', type=int, default=7777, help='Multi-GPU Training Port.')
    parser.add_argument('--chkfreq', type=int, default=100, help='freqency checkpoint')

    # logging
    parser.add_argument('--wandb', action='store_true', help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true', help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, required=False, help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, required=False, help='wandb entity name')
    parser.add_argument('--log_grad', action='store_true', help='toggle to save gradients')
    parser.add_argument('--eval', action='store_false', help='toggle to disable evaluation on in/out test domain')

    # focal loss
    parser.add_argument('--gamma', type=float, default=1, help='gamma for focal loss')

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