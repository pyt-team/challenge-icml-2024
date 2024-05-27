import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool
from modules.models.simplicial.empsn_layer import EMPSNLayer
from typing import Tuple, Dict, List, Literal
from modules.models.model_utils import decompose_batch


class EMPSN(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, max_dim: int) -> None: 
        super().__init__()

        self.max_dim = max_dim
        self.feature_embedding = nn.Linear(in_channels, hidden_channels)

        # Create `num_layers` layers with sum aggregation and SiLU update function
        tmp_dict = {
            "rank_0": {
                f"rank_0": 3,
                f"rank_1": 3
            },
            "rank_1": {
                f"rank_1": 6,
                f"rank_2": 6
            }
        }

        self.layers = nn.ModuleList(
            [EMPSNLayer(hidden_channels, self.max_dim, aggr_func="sum", update_func="silu", aggr_update_func=None, n_inv=tmp_dict) for _ in range(num_layers)]
        )

        # Pre-pooling operation
        self.pre_pool = nn.ModuleDict()
        for rank in range(self.max_dim+1):
            self.pre_pool[f"rank_{rank}"]= nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

        # Post-pooling operation over all dimensions
        # and final classification
        self.post_pool = nn.Sequential(
            nn.Linear((max_dim + 1) * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )


    def forward(self, batch)-> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        batch : torch_geometric.data.BatchData, length=max_rank+1, shape = (n_rank_r_cells, channels)
            Input features on the cells of the simplicial complex.
        Returns
        -------
        Tensor, shape = (1)
            Regression value of the pooling of the graph
        """
        features, edge_index_adj, edge_index_inc, inv_r_r, inv_r_minus_1_r, x_batch = decompose_batch(batch, self.max_dim)

        x = {
            rank: self.feature_embedding(feature) 
            for rank, feature in features.items()
        }
        for layer in self.layers:
            x = layer(x, edge_index_adj, edge_index_inc, inv_r_r, inv_r_minus_1_r)

        # read out
        x = {rank: self.pre_pool[rank](feature) for rank, feature in x.items()}
        x = {rank: global_add_pool(x[rank], batch=x_batch[rank]) for rank, feature in x.items()}
        state = torch.cat(tuple([feature for rank, feature in x.items()]), dim=1)
        out = self.post_pool(state) # Classifier across 19 variables 
        out = torch.squeeze(out)

        return out
    
    def __str__(self):
        return f"EMPSN ({self.type})"


