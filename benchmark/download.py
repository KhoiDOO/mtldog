import wandb
import argparse
import json

def save_json(dct, path):
    with open(path, 'w') as outfile:
        json.dump(dct, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb_prj', type=str, required=False, default='MTLDOG', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, required=False, default='heartbeats', help='wandb entity name')
    parser.add_argument('--run_id', type=str, required=False, default='tzq4560e', help='wandb entity name')

    args = parser.parse_args()

    api = wandb.Api(timeout=100)
    
    run = api.run(f"{args.wandb_entity}/{args.wandb_prj}/{args.run_id}")

    run_summary = run.summary

    history = run.scan_history()

    for row in history:
        print(row)
        break