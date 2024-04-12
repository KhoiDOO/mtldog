from argparse import Namespace
from typing import List, Dict
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity as cossim
from pandas import DataFrame
from tabulate import tabulate
from torch import nn

import pandas as pd
import numpy as np
import json, hashlib, lzma, torch, pickle, gc

def save_json(dct: Dict, path: str) -> None:
    with open(path, 'w') as outfile:
        json.dump(dct, outfile)

def read_json(path: str) -> Dict:
    return json.load(open(path, 'r'))

def save_pickle_xz(dct: Dict, path:str) -> None:
    gc.disable()
    with lzma.open(path, 'wb') as  file:
        pickle.dump(obj=dct, file=file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    gc.enable()

def read_pickle_xz(path:str) -> Dict:
    gc.disable()
    with lzma.open(path, 'rb') as  file:
        dct = pickle.load(file=file)
    file.close()
    gc.enable()
    return dct

def save_pickle(dct: Dict, path:str) -> None:
    with open(path, 'wb') as  file:
        pickle.dump(obj=dct, file=file, protocol=pickle.HIGHEST_PROTOCOL)
    
def read_pickle(path:str) -> None:
    with open(path, 'rb') as  file:
        dct = pickle.load(file=file)
    file.close()
    return dct

def get_hash(args: Namespace) -> str:
    args_str = json.dumps(vars(args), sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash

def lw_eigen(hess: Tensor) -> Tensor:
    return torch.linalg.svdvals(hess if len(hess.size()) > 1 else hess.unsqueeze(0))

def hess_eigen(hess: List[Tensor]) -> Tensor:
    eigens = torch.cat([lw_eigen(_hess).flatten() for _hess in hess])
    return eigens

def single_eigen(vector: Tensor):
    return lw_eigen(vector).flatten()

def cosine_similarity(vectors: List[Tensor]) -> np.array:
    vectors:np.array = torch.stack(vectors).numpy()
    cos_simi_mat = cossim(vectors, vectors)
    return cos_simi_mat

def symmetry_cossine_similarity(lst_a: List[Tensor], lst_b: List[Tensor]) -> List[float]:
    return [cossim(g.flatten().unsqueeze(0).numpy(), h.flatten().unsqueeze(0).numpy()).item() for g,h in zip(lst_a, lst_b)]

def symmetry_dotprod_similarity(lst_a: List[Tensor], lst_b: List[Tensor]) -> List[float]:
    return [np.matmul(g.flatten().unsqueeze(0).numpy(), h.flatten().unsqueeze(0).numpy().T).item() for g,h in zip(lst_a, lst_b)]

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

def lwflat(vectors: List[Tensor]):
    return torch.cat([x.flatten() for x in vectors])

def lwflatusq(vectors: List[Tensor]):
    return torch.cat([x.flatten() for x in vectors]).unsqueeze(0)

def usq(vector: Tensor):
    return vector.unsqueeze(0)

def norm2(vector: Tensor):
    return vector.norm(p=2).item()

def distinct_pairs(lst: List):
    _lst = []
    for i, x in enumerate(lst):
        for y in lst[i+1:]:
            _lst.append((x, y))
    return _lst

def show_log(mean_log: Dict, round: int, args: Namespace):
    print(f"ROUND: {round}\n", "{:<50} {:<50}\n".format('KEY', 'VALUE'), "*"*100)

    share_gradlw_table = {}
    heads_gradlw_table = {tk : {} for tk in args.tkss}

    for key, value in mean_log.items():
        if not isinstance(value, Tensor) and not (isinstance(value, DataFrame)):
            print("{:<50} {:<50}\n".format(key, value), "-"*100)
        
        elif isinstance(value, Tensor) and args.extend_verbose:
            print("{:<50} {:<50}\n".format(key, "x".join([str(x) for x in list(value.size())])), "-"*100)

        elif isinstance(value, DataFrame) and args.extend_verbose:
            if 'grad' in key:
                if 'lw' in key:
                    if 'share' in key:
                        share_gradlw_table[key] = tabulate(value, headers='keys', tablefmt='psql')
                    else:
                        heads_gradlw_table[key.split('-')[2]][key] = tabulate(value, headers='keys', tablefmt='psql')
        elif isinstance(value, list):
            print(key, 'is list')
        
    if args.extend_verbose:
        print(table_cascade(share_gradlw_table))
        for tk in args.tkss:
            print(table_cascade(heads_gradlw_table[tk]))

def table_cascade(table_dict: Dict[str, str]):

    master_headers = list(table_dict.keys())

    tables = [str(x).splitlines() for x in list(table_dict.values())]

    master_table = tabulate([list(item) for item in zip(*tables)], master_headers, tablefmt="simple")

    return master_table

def preprocess_analysis(hess_dict: Dict[str, Dict[str, Dict[str, Dict[str, List[Tensor]]]]] | None = None, 
                         sol_grad_share: Tensor | None = None, sol_grad_head: Dict[str, Tensor] | None = None,
                         args: Namespace | None = None) -> Dict[str, Tensor | DataFrame| float]:

    assert args is not None
    
    log_dict = {}

    if hess_dict is None:
        return log_dict

    # add tensor
    for dm, dm_dict in hess_dict.items():
        for tk, tk_dict in dm_dict.items():

            share_dict = tk_dict["share"]
            head_dict = tk_dict["head"]

            share_dict['grad_flat'] = lwflat(share_dict['grad'])
            share_dict['hess_flat'] = lwflat(share_dict['hess'])

            head_dict['grad_flat'] = lwflat(head_dict['grad'])
            head_dict['hess_flat'] = lwflat(head_dict['hess'])

            if args.quant:
                for ln, grad, hess in zip(share_dict['name'], share_dict['grad'], share_dict['hess']):
                    
                    log_dict[f"grad-share-{dm}-{tk}/{ln}-vec"] = grad
                    log_dict[f"hess-share-{dm}-{tk}/{ln}-vec"] = hess
                    temp_eigen = lw_eigen(hess)
                    log_dict[f"hess-share-eigen-{dm}-{tk}/{ln}-vec"] = temp_eigen

                for ln, grad, hess in zip(head_dict['name'], head_dict['grad'], head_dict['hess']):
                    log_dict[f"grad-head-{dm}-{tk}/{ln}-vec"] = grad
                    log_dict[f"hess-head-{dm}-{tk}/{ln}-vec"] = hess
                    temp_eigen = lw_eigen(hess)
                    log_dict[f"hess-head-eigen-{dm}-{tk}/{ln}-vec"] = temp_eigen
                
                log_dict[f'grad-share/{dm}-{tk}-vec'] = share_dict['grad_flat']
                log_dict[f'grad-head/{dm}-{tk}-vec'] = head_dict['grad_flat']
            
            log_dict[f"hess-share-eigen/{dm}-{tk}-norm"] = norm2(single_eigen(share_dict['hess_flat']))
            log_dict[f"hess-head-eigen/{dm}-{tk}-norm"] = norm2(single_eigen(head_dict['hess_flat']))
            
            log_dict[f"grad-share/{dm}-{tk}-norm"] = norm2(share_dict['grad_flat'])
            log_dict[f"grad-head/{dm}-{tk}-norm"] = norm2(head_dict['grad_flat'])
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    task_pairs = distinct_pairs(args.tkss)
    dm_pairs = distinct_pairs(list(hess_dict.keys()))

    for dm, dm_dict in hess_dict.items():
        for tk_i, tk_j in task_pairs:

            tk_dict_i = dm_dict[tk_i]
            tk_dict_j = dm_dict[tk_j]

            cosine = cos(usq(tk_dict_i['share']['grad_flat']), usq(tk_dict_j['share']['grad_flat'])).item()
            log_dict[f"grad-share-cosine/{dm}-{tk_i}-vs-{tk_j}"] = cosine
    
    for dm_i, dm_j in dm_pairs:
        dm_dict_i = hess_dict[dm_i]
        dm_dict_j = hess_dict[dm_j]

        for tk in args.tkss:

            tk_dm_dict_i = dm_dict_i[tk]
            tk_dm_dict_j = dm_dict_j[tk]

            cosine = cos(usq(tk_dm_dict_i['share']['grad_flat']), usq(tk_dm_dict_j['share']['grad_flat'])).item()
            log_dict[f"grad-share-cosine/{dm_i}-vs-{dm_j}-{tk}"] = cosine

            cosine = cos(usq(tk_dm_dict_i['head']['grad_flat']), usq(tk_dm_dict_j['head']['grad_flat'])).item()
            log_dict[f"grad-head-cosine/{dm_i}-vs-{dm_j}-{tk}"] = cosine
    
    if sol_grad_share:
        log_dict[f"grad-sol/share-norm"] = norm2(sol_grad_share)
        if args.quant:
            log_dict[f"grad-sol/share-vec"] = sol_grad_share

        for dm, dm_dict in hess_dict.items():
            for tk, tk_dict in dm_dict.items():
                share_dict = tk_dict["share"]

                cosine = cos(usq(share_dict['grad_flat']), usq(sol_grad_share)).item()
                log_dict[f"grad-sol-cosine/share-{dm}-{tk}"] = cosine
    
    if sol_grad_head:
        for tk, vector in sol_grad_head.items():
            log_dict[f"grad-sol/head-{tk}-norm"] = norm2(vector)
            if args.quant:
                log_dict[f"grad-sol/head-{tk}-vec"] = vector
        
        for dm, dm_dict in hess_dict.items():
            for tk, tk_dict in dm_dict.items():
                head_dict = tk_dict["head"]

                cosine = cos(usq(head_dict['grad_flat']), usq(sol_grad_share[tk])).item()
                log_dict[f"grad-sol-cosine/head-{dm}-{tk}"] = cosine

    if args.quant:
        share_cos_dct = {'layer-name' : hess_dict[list(hess_dict.keys())[0]][args.tkss[0]]['share']['name']}
        share_dot_dct = {'layer-name' : hess_dict[list(hess_dict.keys())[0]][args.tkss[0]]['share']['name']}
        for dm, dm_dict in hess_dict.items():
            for tk_i, tk_j in task_pairs:

                tk_dict_i = dm_dict[tk_i]
                tk_dict_j = dm_dict[tk_j]

                share_cos_dct[f'{dm}/{tk_i}-vs-{tk_j}'] = symmetry_cossine_similarity(tk_dict_i['share']['grad'], tk_dict_j['share']['grad'])
            
            for tk in args.tkss:
                tk_dict = dm_dict[tk]
                share_dot_dct[f'{dm}/{tk}'] = symmetry_dotprod_similarity(tk_dict['share']['grad'], tk_dict['share']['grad'])
        
        for dm_i, dm_j in dm_pairs:

            dm_dict_i = hess_dict[dm_i]
            dm_dict_j = hess_dict[dm_j]

            for tk in args.tkss:

                tk_dm_dict_i = dm_dict_i[tk]
                tk_dm_dict_j = dm_dict_j[tk]

                share_cos_dct[f'{dm_i}-vs-{dm_j}/{tk}'] = symmetry_cossine_similarity(tk_dm_dict_i['share']['grad'], tk_dm_dict_j['share']['grad'])

        log_dict[f'grad-share-cos-lw-tab'] = pd.DataFrame(share_cos_dct)
        log_dict[f'grad-share-dot-lw-tab'] = pd.DataFrame(share_dot_dct)

        for tk in args.tkss:
            head_cos_dict = {'layer-name' : hess_dict[list(hess_dict.keys())[0]][tk]['head']['name']}
            head_dot_dict = {'layer-name' : hess_dict[list(hess_dict.keys())[0]][tk]['head']['name']}
            for dm_i, dm_j in dm_pairs:

                tk_dm_dict_i = hess_dict[dm_i][tk]
                tk_dm_dict_j = hess_dict[dm_j][tk]

                head_cos_dict[f'{dm_i}-vs-{dm_j}/{tk}'] = symmetry_cossine_similarity(tk_dm_dict_i['head']['grad'], tk_dm_dict_j['head']['grad'])
            
            for dm, dm_dict in hess_dict.items():

                tk_dm_dict = hess_dict[dm][tk]
                
                head_dot_dict[f'{dm}/{tk}'] = symmetry_dotprod_similarity(tk_dm_dict['head']['grad'], tk_dm_dict['head']['grad'])
        
            log_dict[f'grad-heads-{tk}-cos-lw-tab'] = pd.DataFrame(head_cos_dict)
            log_dict[f'grad-heads-{tk}-dot-lw-tab'] = pd.DataFrame(head_dot_dict)

    return log_dict