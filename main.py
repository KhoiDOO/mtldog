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

    # training
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    
