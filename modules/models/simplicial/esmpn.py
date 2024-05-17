import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.nn import global_add_pool
from typing import Tuple, Dict, List, Literal
from utils import compute_invariants_3d


class EMPSN(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, num_input: int, num_hidden: int, num_out: int, num_layers: int, max_com: str, max_dim: int) -> None: 
        super().__init__()

        self.max_dim = max_dim
        self.feature_embedding = nn.Linear(num_input, num_hidden)

        # Create `num_layers` layers with sum aggregation and SiLU update function
        self.layers = nn.ModuleList(
            [EMPSNLayer(num_hidden, self.max_dim, aggr_func="sum", update_func="silu") for _ in range(num_layers)]
        )

        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.post_pool = nn.Sequential(
            nn.Sequential(nn.Linear((max_dim + 1) * num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_out))
        )

    def forward(self, features, adjacencies, incidence, invariances_r_r_minus_1, invariances_r_r_minus_1) -> Tensor:
        
        for layer in self.layers:
            x = layer(x, adjacencies, incidence, invariances_r_r, invariances_r_r_minus_1)

        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        x = {dim: global_add_pool(x[dim], x_batch[dim]) for dim, feature in x.items()}
        state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        out = self.post_pool(state) #classifier across 19 variables 
        out = torch.squeeze(out)

        return out
    
    def __str__(self):
        return f"EMPSN ({self.type})"


