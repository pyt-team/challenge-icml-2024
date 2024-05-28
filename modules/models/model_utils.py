from typing import Dict, Tuple

from modules.data.utils.utils import compute_invariance_r_to_r, compute_invariance_r_minus_1_to_r

def get_invariances(batch, max_rank):
    if 'inv_same_0' in batch.keys():
        inv_r_r = {
            f"rank_{i}": batch[f'inv_same_{i}'] for i in range(0, max_rank)
        }
        inv_r_minus_1_r = {
            f"rank_{i}": batch[f'inv_low_high_{i}'] for i in range(1, max_rank)
        }
    else:
        adj_dict = {}
        feat_ind = {}
        inc_dict = {}

        for r in range(max_rank+1):
            adj_dict[r] = batch[f'adjacency_{r}']
            feat_ind[r] = batch[f'x_idx_{r}']

        for r in range(max_rank):
            inc_dict[r] = batch[f'incidence_{r+1}']

        inv_same_dict = compute_invariance_r_to_r(feat_ind, batch['pos'], adj_dict)
        inv_low_high_dict = compute_invariance_r_minus_1_to_r(feat_ind, batch['pos'], inc_dict)

        inv_r_r = {}
        inv_r_minus_1_r = {}

        # Set invariances in data
        # Fix for the mismatch in computing invariances above
        for r in range(0, max_rank+1):
            if r != max_rank:
                inv_r_r[f'rank_{r}'] = inv_same_dict[r]
            if r > 0:
                inv_r_minus_1_r[f'rank_{r}'] = inv_low_high_dict[r-1]
    return inv_r_r, inv_r_minus_1_r

def decompose_batch(batch, max_rank) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    inv_r_r, inv_r_minus_1_r = get_invariances(batch, max_rank)
    features = {
        f"rank_{i}": batch[f'x_{i}'] for i in range(max_rank+ 1)
    }
    edge_index_adj = {
        f"rank_{i}": batch[f'adjacency_{i}'] for i in range(0, max_rank + 1)
    }
    edge_index_inc = {
        f"rank_{i}": batch[f'incidence_{i}'] for i in range(0, max_rank + 1)
    }
    x_batch = {
        f"rank_{i}": batch[f'x_{i}_batch'] for i in range(max_rank + 1)
    }

    return features, edge_index_adj, edge_index_inc, inv_r_r, inv_r_minus_1_r, x_batch