from argparse import Namespace
from typing import List, Dict
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity as cossim
from pandas import DataFrame
from tabulate import tabulate
from numpy.linalg import svd

import pandas as pd
import numpy as np
import json, hashlib, lzma, torch, pickle, gc

def save_json(dct: Dict, path: str) -> None:
    with open(path, 'w') as outfile:
        json.dump(dct, outfile)

def read_json(path: str) -> Dict:
    return json.load(open(path, 'r'))

def save_pickle(dct: Dict, path:str) -> None:
    gc.disable()
    with lzma.open(path, 'wb') as  file:
        pickle.dump(obj=dct, file=file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    gc.enable()

def read_pickle(path:str) -> Dict:
    gc.disable()
    with lzma.open(path, 'rb') as  file:
        dct = pickle.load(file=file)
    file.close()
    gc.enable()
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

def distinct_pairs(lst: List):
    _lst = []
    for i, x in enumerate(lst):
        for y in lst[i+1:]:
            _lst.append((x, y))
    return _lst

def show_log(mean_log: Dict, round: int, args: Namespace):
    print(f"ROUND: {round}\n", "{:<50} {:<50}\n".format('KEY', 'VALUE'), "*"*100)

    share_grad_table = {}
    heads_grad_table = {}
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
                else:
                    if 'share' in key:
                        share_grad_table[key] = tabulate(value, headers='keys', tablefmt='psql')
                    else:
                        heads_grad_table[key] = tabulate(value, headers='keys', tablefmt='psql')
        elif isinstance(value, list):
            print(key, 'is list')
        
    if args.extend_verbose:
        print(table_cascade(share_grad_table))
        print(table_cascade(heads_grad_table))
        print(table_cascade(share_gradlw_table))
        for tk in args.tkss:
            print(table_cascade(heads_gradlw_table[tk]))

def table_cascade(table_dict: Dict[str, str]):

    master_headers = list(table_dict.keys())

    tables = [str(x).splitlines() for x in list(table_dict.values())]

    master_table = tabulate([list(item) for item in zip(*tables)], master_headers, tablefmt="simple")

    return master_table

def preprocess_grad_train(grad_dict: Dict[str, Dict[str, Tensor | Dict[str, Tensor]]], sol_grad_share: Tensor, sol_grad_head: Dict[str, Tensor], args: Namespace) -> Dict[str, Tensor | DataFrame| float]:
    
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

                # main_grad_dict[f'grad-share/{dmtxt}-{tk}-vec'] = share_grad_dict[dmtxt][tkidx]
                main_grad_dict[f'grad-share/{dmtxt}-{tk}-norm'] = share_grad_dict[dmtxt][tkidx].norm(p=2).item()
        
        main_grad_dict['grad-share-cos-tab'] = cosine_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)
        main_grad_dict['grad-share-dot-tab'] = dot_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)

        if sol_grad_share is not None:
            main_grad_dict[f'grad-share/sol-vec'] = sol_grad_share
            share_grad_keys.append('sol-grad')
            share_grad_vectors.append(sol_grad_share)

            main_grad_dict['grad-share-sol-cos-tab'] = cosine_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)
            main_grad_dict['grad-share-sol-dot-tab'] = dot_dataframe(keys=share_grad_keys, vectors=share_grad_vectors)

        for _, tk in enumerate(args.tkss):
            head_grad_keys = []
            head_grad_vectors = []
            for dmtxt in head_grad_dict:
                head_grad_keys.append(f'{dmtxt}-{tk}')
                head_grad_vectors.append(head_grad_dict[dmtxt][tk])
            
                # main_grad_dict[f'grad-heads/{dmtxt}-{tk}-vec'] = head_grad_dict[dmtxt][tk]
                main_grad_dict[f'grad-heads/{dmtxt}-{tk}-norm'] = head_grad_dict[dmtxt][tk].norm(p=2).item()

            main_grad_dict[f'grad-heads-{tk}-cos-tab'] = cosine_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
            main_grad_dict[f'grad-heads-{tk}-dot-tab'] = dot_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)

            if sol_grad_head is not None:
                main_grad_dict[f'grad-head/sol-{tk}-vec'] = sol_grad_head[tk]
                head_grad_keys.append(f'sol-{tk}-vec')
                head_grad_vectors.append(sol_grad_head[tk])

                main_grad_dict[f'grad-heads-sol-{tk}-cos-tab'] = cosine_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
                main_grad_dict[f'grad-heads-sol-{tk}-dot-tab'] = dot_dataframe(keys=head_grad_keys, vectors=head_grad_vectors)
    
    return main_grad_dict

def preprocess_grad_hess_adv(hess_dict: Dict[str, Dict[str, Dict[str, Dict[str, List[Tensor]]]]], args: Namespace) -> Dict[str, Tensor | DataFrame| float]:
    
    grad_hess_dict = {}

    if hess_dict is None:
        return grad_hess_dict

    # add tensor
    for dm, dm_dict in hess_dict.items():
        for tk, tk_dict in dm_dict.items():
            share_dict = tk_dict["share"]
            head_dict = tk_dict["head"]

            for ln, grad, hess in zip(share_dict['name'], share_dict['grad'], share_dict['hess']):
                
                grad_hess_dict[f"grad-share-{dm}-{tk}/{ln}-vec"] = grad
                grad_hess_dict[f"hess-share-{dm}-{tk}/{ln}-vec"] = hess
                temp_eigen = lw_eigen(hess)
                grad_hess_dict[f"hess-share-eigen-{dm}-{tk}/{ln}-vec"] = temp_eigen
            
            # grad_hess_dict[f"hess-share-eigen/{dm}-{tk}-vec"] = hess_eigen(share_dict['hess'])
            grad_hess_dict[f"hess-share-eigen/{dm}-{tk}-norm"] = hess_eigen(share_dict['hess']).norm(p=2).item()

            for ln, grad, hess in zip(head_dict['name'], head_dict['grad'], head_dict['hess']):
                grad_hess_dict[f"grad-head-{dm}-{tk}/{ln}-vec"] = grad
                grad_hess_dict[f"hess-head-{dm}-{tk}/{ln}-vec"] = hess
                temp_eigen = lw_eigen(hess)
                grad_hess_dict[f"hess-head-eigen-{dm}-{tk}/{ln}-vec"] = temp_eigen
            
            # grad_hess_dict[f"hess-head-eigen/{dm}-{tk}-vec"] = hess_eigen(share_dict['hess'])
            grad_hess_dict[f"hess-head-eigen/{dm}-{tk}-norm"] = hess_eigen(share_dict['hess']).norm(p=2).item()

    # layer-wise task-wise cosine matrix
    task_pairs = distinct_pairs(args.tkss)
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

    dm_pairs = distinct_pairs(list(hess_dict.keys()))
    
    for dm_i, dm_j in dm_pairs:

        dm_dict_i = hess_dict[dm_i]
        dm_dict_j = hess_dict[dm_j]

        for tk in args.tkss:

            tk_dm_dict_i = dm_dict_i[tk]
            tk_dm_dict_j = dm_dict_j[tk]

            share_cos_dct[f'{dm_i}-vs-{dm_j}/{tk}'] = symmetry_cossine_similarity(tk_dm_dict_i['share']['grad'], tk_dm_dict_j['share']['grad'])

    grad_hess_dict[f'grad-share-cos-lw-tab'] = pd.DataFrame(share_cos_dct)
    grad_hess_dict[f'grad-share-dot-lw-tab'] = pd.DataFrame(share_dot_dct)

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
    
        grad_hess_dict[f'grad-heads-{tk}-cos-lw-tab'] = pd.DataFrame(head_cos_dict)
        grad_hess_dict[f'grad-heads-{tk}-dot-lw-tab'] = pd.DataFrame(head_dot_dict)

    return grad_hess_dict