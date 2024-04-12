import os
import wandb
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_grad_share(run, args):
    history = run.scan_history()
    config = run.config

    for x in history:
        print(x['grad-share-cos-tab'])
        exit()

def draw_grad_head(run, args):
    pass


if __name__ == "__main__":

    save_dir = os.path.dirname(os.path.abspath(__file__)) + "/single_grad_cosine"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--rid', type=str, required=True, help='run id')

    args = parser.parse_args()

    api = wandb.Api()

    run = api.run(f"heartbeats/MTLDOG/{args.rid}")
    
    if run.config['grad']:
        pass
    else:
        raise Exception(f"The run experiment contains no quantitative analysis on gradient")

    draw_grad_share(run=run, args=args)

