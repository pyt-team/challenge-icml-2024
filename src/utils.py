import numpy as np
import random
import torch
import os

from typing import Dict, Tuple

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

def decompose_batch(args, batch) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    features = {
        f"rank_{i}": batch[f'x_{i}'].to(args.device) for i in range(args.dim + 1)
    }
    edge_index_adj = {
        f"rank_{i}": batch[f'adjacency_{i}'].to_dense().nonzero().t().contiguous().to(args.device) for i in range(0, args.dim + 1)
    }

    edge_index_inc = {
        f"rank_{i}": batch[f'incidence_{i}'].to_dense().nonzero().t().contiguous().to(args.device) for i in range(0, args.dim + 1)
    }
    inv_r_r = {
        f"rank_{i}": batch[f'inv_same_{i}'].to(args.device) for i in range(0, args.dim)
    }
    inv_r_minus_1_r = {
        f"rank_{i}": batch[f'inv_low_high_{i}'].to(args.device) for i in range(1, args.dim+1)
    }
    x_batch = {
        f"rank_{i}": batch[f'x_{i}_batch'] for i in range(args.dim + 1)
    }
    return features, edge_index_adj, edge_index_inc, inv_r_r, inv_r_minus_1_r, x_batch
