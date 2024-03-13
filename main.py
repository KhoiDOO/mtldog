"""main.py - CLI Interaction"""

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='MTLDOG - Domain Generalization for Multi-task Learning')

    # dataset
    parser.add_argument('--ds', type=str, required=True, help='dataset in use')
    parser.add_argument('--dt', type=str, required=False, help='root data directory')
    parser.add_argument('--bs', type=int, required=True, default=64, help='batch size')
    parser.add_argument('--wk', type=int, required=True, help='no. ds worker')
    parser.add_argument('--pm', action='store_true', help='toggle to use pin memory')

    # domain
    parser.add_argument('--trdms', type=int, nargs='+', required=True, default=[0], help='list of domain used in training')
    
    # task
    parser.add_argument('--trtks', type=str, nargs='+', required=True, help='list of tasks used in training')

    # method

    # model
    parser.add_argument('--at', type=str, required=True, help='archiecture type (i.e. ae, unet)')
    parser.add_argument('--bb', type=str, required=True, help='backbone type (i.e. ae, base, resnet18)')

    # training
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--tm', type=str, default='sup', help='training mode (i.e. supervised)')
    parser.add_argument('--dvids', type=str, nargs='+', default=[0], help='list of device used in training')

    args = parser.parse_args()