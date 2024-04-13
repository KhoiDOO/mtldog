from glob import glob
from alive_progress import alive_it
from typing import Dict

import os
import pandas as pd
import json

DOMAINS = ['alarm', 'calling', 'event', 'messaging', 'music', 'news', 'people', 'recipes', 'reminder', 'timer', 'weather']
LANGUAGES = ['de', 'en', 'es', 'fr', 'hi', 'th']


def save_json(dct: Dict, path: str) -> None:
    with open(path, 'w') as outfile:
        json.dump(dct, outfile)

def read_json(path: str) -> Dict:
    return json.load(open(path, 'r'))

def preprocess_mtop(data_dir: str, proc_dir:str):

    total_error = 0

    if not os.path.exists(data_dir):
        raise ValueError(f'data_dir: {data_dir} does not exist')

    for dm in DOMAINS:
        dm_dir = proc_dir + f"/{dm}"

        for lan in LANGUAGES:
            lan_dir = dm_dir + f"/{lan}"

            os.makedirs(lan_dir, exist_ok=True)

    txt_files = glob(data_dir + "/*/*.txt")

    for txt in alive_it(txt_files):

        txt_split = txt.split("/")

        lan, split = txt_split[-2], txt_split[-1].replace(".txt", "")

        data = pd.read_csv(txt, sep='\t')

        for domain in DOMAINS:
            dm_data = data[data.iloc[:, 4] == domain].iloc[:, [0, 3, 4, 7]]

            ids, uter, dm = dm_data.iloc[:, 0].tolist(), dm_data.iloc[:, 1].tolist(), dm_data.iloc[:, 2].to_list()

            # token_dcts = [json.loads(x) for x in dm_data.iloc[:, 3].tolist()]

            token_dcts = []

            for x in dm_data.iloc[:, 3].tolist():
                try:
                    token_dcts.append(json.loads(fr"{x}"))
                except:
                    total_error += 1

            tokens, tokenspans = [x['tokens'] for x in token_dcts], [x['tokenSpans'] for x in token_dcts]

            dct = {"ids" : ids, "uter" : uter, "dm" : dm, "tokens" : tokens, "tokenspans" : tokenspans}

            save_path = proc_dir + f"/{domain}/{lan}/{split}.json"

            save_json(dct=dct, path=save_path)
    
    return total_error 

def check(proc_dir:str):

    json_files = glob(proc_dir + f"/*/*/*.json")

    for js in json_files:
        split = js.split("/")

        dm, lan, sp = split[-3], split[-2], split[-1].replace(".json", "")

        dct = read_json(js)

        print(f"DOMAIN: {dm} - LAN: {lan} - SP: {sp}", end="\n\t")
        print(" - ".join([f"{key.upper()} : {len(value)}" for key, value in dct.items()]))
        print(" - ".join([f"{key.upper()} : {value[0]}" for key, value in dct.items()]))


if __name__ == "__main__":
    data_dir = "/media/mountHDD3/data_storage/mtop/mtop"
    proc_dir = "/media/mountHDD3/data_storage/mtop/mtop_proc"

    total_error = preprocess_mtop(data_dir=data_dir, proc_dir=proc_dir)

    print(f"Total Error (JSON Decoding) : {total_error}")

    check(proc_dir=proc_dir)