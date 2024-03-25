from argparse import Namespace
from typing import List
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity as cossim

import pandas as pd
import numpy as np
import json, hashlib
import torch

def save_json(dct, path):
    with open(path, 'w') as outfile:
        json.dump(dct, outfile)

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
    return pd.DataFrame(data=mat, columns=keys)

def cosine_dataframe(keys: List[str], vectors: List[Tensor]):
    return matrix_dataframe(keys=keys, mat=cosine_similarity(vectors=vectors))

def dot_dataframe(keys: List[str], vectors: List[Tensor]):
    return matrix_dataframe(keys=keys, mat=dot_similarity(vectors=vectors))