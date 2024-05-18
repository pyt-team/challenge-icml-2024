import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.nn import global_add_pool
from modules.models.simplicial.esmpn_layer import EMPSNLayer
from modules.base.econv import EConv
from typing import Tuple, Dict, List, Literal
from utils import compute_invariants_3d


class EMPSN(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, max_com: str, max_dim: int) -> None: 
        super().__init__()

        self.max_dim = max_dim
        self.feature_embedding = nn.Linear(in_channels, hidden_channels)

        # Create `num_layers` layers with sum aggregation and SiLU update function
        self.layers = nn.ModuleList(
            [EMPSNLayer(hidden_channels, self.max_dim, aggr_func="sum", update_func="silu") for _ in range(num_layers)]
        )

        self.pre_pool = nn.ModuleList(
            [

                EConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    update_func="silu"
                ),
                EConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    update_func=None
                ),
            ]
            
            for dim in range(self.max_dim+1)
        )

        self.post_pool = nn.ModuleList(
            [

                EConv(
                    in_channels=(max_dim+1) * hidden_channels,
                    out_channels=hidden_channels,
                    update_func="silu"
                ),
                EConv(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    update_func=None
                ),
            ]
            
            for dim in range(self.max_dim+1)
        )

    def forward(self, features, adjacencies, incidence, invariances_r_r, invariances_r_r_minus_1) -> Tensor:

        x = features 
        for layer in self.layers:
            x = layer(x, adjacencies, incidence, invariances_r_r, invariances_r_r_minus_1)

        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        # TODO how to batch
        x = {dim: global_add_pool(x[dim]) for dim, feature in x.items()}
        state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        out = self.post_pool(state) # Classifier across 19 variables 
        out = torch.squeeze(out)

        return out
    
    def __str__(self):
        return f"EMPSN ({self.type})"


