from argparse import Namespace
from typing import List, Dict
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity as cossim
from pandas import DataFrame
from tabulate import tabulate

import pandas as pd
import numpy as np
import json, hashlib
import torch
import pickle


SPACE = 50


def save_json(dct: Dict, path: str) -> None:
    with open(path, 'w') as outfile:
        json.dump(dct, outfile)

def read_json(path: str) -> Dict:
    return json.load(open(path, 'r'))

def save_pickle(dct: Dict, path:str) -> None:
    with open(path, 'wb') as  file:
        pickle.dump(obj=dct, file=file)
    file.close()

def read_pickle(path:str) -> Dict:
    with open(path, 'rb') as  file:
        dct = pickle.load(file=file)
    file.close()
    return dct

def get_hash(args: Namespace) -> str:
    args_str = json.dumps(vars(args), sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash

def cosine_similarity(vectors: List[Tensor]) -> np.array:
    vectors:np.array = torch.stack(vectors).numpy()
    cos_simi_mat = cossim(vectors, vectors)
    return cos_simi_mat

def dot_similarity(vectors: List[Tensor]) -> np.array:
    vectors:np.array = torch.stack(vectors).numpy()
    dot_simi_mat = np.matmul(vectors, vectors.T)
    return dot_simi_mat

def matrix_dataframe(keys: List[str], mat:np.array):
    return pd.DataFrame(data=mat, columns=keys, index=keys)

def cosine_dataframe(keys: List[str], vectors: List[Tensor]):
    return matrix_dataframe(keys=keys, mat=cosine_similarity(vectors=vectors))

def dot_dataframe(keys: List[str], vectors: List[Tensor]):
    return matrix_dataframe(keys=keys, mat=dot_similarity(vectors=vectors))

def show_log(mean_log: Dict, round: int):
    print(f"ROUND: {round}")
    print("{:<70} {:<70}".format('KEY', 'VALUE'))
    print("*"*140)

    share_table = {}
    heads_table = {}

    for key, value in mean_log.items():
        if not isinstance(value, Tensor) and not (isinstance(value, DataFrame)):
            print("{:<50} {:<50}".format(key, value))
            print("-"*140)

        elif isinstance(value, DataFrame):
            if 'share' in key:
                share_table[key] = tabulate(value, headers='keys', tablefmt='psql')
            else:
                heads_table[key] = tabulate(value, headers='keys', tablefmt='psql')
    
    print(table_cascade(share_table))
    print(table_cascade(heads_table))
    

def table_cascade(table_dict: Dict[str, str]):

    master_headers = list(table_dict.keys())

    tables = [str(x).splitlines() for x in list(table_dict.values())]

    master_table = tabulate([list(item) for item in zip(*tables)], master_headers, tablefmt="simple")

    return master_table

def preprocess_grad_train(grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]], sol_grad_share: Tensor, sol_grad_head: Dict[str, Tensor], args: Namespace) -> Dict[str, Tensor | DataFrame]:
    
    main_grad_dict = {}
    if grad_dict is not None:
        share_grad_dict = {dmtxt : grad_dict[dmtxt]['share'] for dmtxt in grad_dict}
        head_grad_dict = {dmtxt : grad_dict[dmtxt]['heads'] for dmtxt in grad_dict}

        share_grad_keys = []
        share_grad_vectors = []
        for dmtxt in share_grad_dict:
            for tkidx, tk in enumerate(args.tkss):
                share_grad_keys.append(f'{dmtxt}-{tk}')
                share_grad_vectors.append(share_grad_dict[dmtxt][tkidx])

                main_grad_dict[f'grad-share-{dmtxt}-{tk}-vec'] = share_grad_dict[dmtxt][tkidx]
        
        main_grad_dict['grad-share-cos-mat'] = cosine_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)
        main_grad_dict['grad-share-dot-mat'] = dot_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)

        if sol_grad_share is not None:
            main_grad_dict[f'sol-grad-share-vec'] = sol_grad_share
            share_grad_keys.append('sol-grad')
            share_grad_vectors.append(sol_grad_share)

            main_grad_dict['sol-grad-share-cos-mat'] = cosine_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)
            main_grad_dict['sol-grad-share-dot-mat'] = dot_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)

        for _, tk in enumerate(args.tkss):
            head_grad_keys = []
            head_grad_vectors = []
            for dmtxt in head_grad_dict:
                head_grad_keys.append(f'{dmtxt}-{tk}')
                head_grad_vectors.append(head_grad_dict[dmtxt][tk])
            
                main_grad_dict[f'grad-heads-{dmtxt}-{tk}-vec'] = head_grad_dict[dmtxt][tk]

            main_grad_dict[f'grad-heads-{tk}-cos-mat'] = cosine_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
            main_grad_dict[f'grad-heads-{tk}-dot-mat'] = dot_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)

            if sol_grad_head is not None:
                main_grad_dict[f'sol-grad-head-{tk}-vec'] = sol_grad_head[tk]
                head_grad_keys.append(f'sol-grad-head-{tk}-vec')
                head_grad_vectors.append(sol_grad_head[tk])

                main_grad_dict[f'sol-grad-heads-{tk}-cos-mat'] = cosine_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
                main_grad_dict[f'sol-grad-heads-{tk}-dot-mat'] = dot_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
    
    return main_grad_dict

def preprocess_grad_hess_adv(hess_dict: Dict[str, Dict[str, Dict[str, Dict[str, List[Tensor]]]]], args: Namespace):
    
    grad_hess_dict = {}

    # add tensor
    for dm, dm_dict in hess_dict.items():
        for tk, tk_dict in dm_dict.items():
            share_dict = tk_dict["share"]
            heads_dict = tk_dict["head"]

            for _tk in args.tkss:
                if _tk == tk:
                    continue
                key = f"grad-lw-{dm}-{tk}-vs-{_tk}-mat"

                grad_flat = [g.flatten() for g in share_dict['grad']]