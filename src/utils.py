import numpy as np
import random
import torch
import os

from typing import Dict, Tuple
from modules.transforms.data_manipulations.manipulations import compute_invariance_r_to_r, compute_invariance_r_minus_1_to_r

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_invariances(args, batch):
    if args.pre_proc:
        inv_r_r = {
            f"rank_{i}": batch[f'inv_same_{i}'].to(args.device) for i in range(0, args.dim)
        }
        inv_r_minus_1_r = {
            f"rank_{i}": batch[f'inv_low_high_{i}'].to(args.device) for i in range(1, args.dim+1)
        }
    else:
        adj_dict = {}
        feat_ind = {}
        inc_dict = {}

        for r in range(args.dim+1):
            adj_dict[r] = batch[f'adjacency_{r}']
            feat_ind[r] = batch[f'x_idx_{r}']

        for r in range(args.dim):
            inc_dict[r] = batch[f'incidence_{r+1}']

        inv_same_dict = compute_invariance_r_to_r(feat_ind, batch['pos'], adj_dict)
        inv_low_high_dict = compute_invariance_r_minus_1_to_r(feat_ind, batch['pos'], inc_dict)

        inv_r_r = {}
        inv_r_minus_1_r = {}

        # Set invariances in data
        # Fix for the mismatch in computing invariances above
        for r in range(0, args.dim+1):
            if r != args.dim:
                inv_r_r[f'rank_{r}'] = inv_same_dict[r]
            if r > 0:
                inv_r_minus_1_r[f'rank_{r}'] = inv_low_high_dict[r-1]
    return inv_r_r, inv_r_minus_1_r

def decompose_batch(args, batch) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    inv_r_r, inv_r_minus_1_r = get_invariances(args, batch)
    features = {
        f"rank_{i}": batch[f'x_{i}'].to(args.device) for i in range(args.dim + 1)
    }
    edge_index_adj = {
        f"rank_{i}": batch[f'adjacency_{i}'].to_dense().nonzero().t().contiguous().to(args.device) for i in range(0, args.dim + 1)
    }
    edge_index_inc = {
        f"rank_{i}": batch[f'incidence_{i}'].to_dense().nonzero().t().contiguous().to(args.device) for i in range(0, args.dim + 1)
    }
    x_batch = {
        f"rank_{i}": batch[f'x_{i}_batch'].to(args.device) for i in range(args.dim + 1)
    }

    return features, edge_index_adj, edge_index_inc, inv_r_r, inv_r_minus_1_r, x_batch
