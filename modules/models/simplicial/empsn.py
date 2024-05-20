import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.nn import global_add_pool
from modules.models.simplicial.empsn_layer import EMPSNLayer
from modules.base.econv import EConv
from typing import Tuple, Dict, List, Literal
from utils import compute_invariants_3d


class EMPSN(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, max_dim: int) -> None: 
        super().__init__()

        self.max_dim = max_dim
        self.feature_embedding = nn.Linear(in_channels, hidden_channels)

        # Create `num_layers` layers with sum aggregation and SiLU update function

        self.layers = nn.ModuleList(
            [EMPSNLayer(hidden_channels, self.max_dim, aggr_func="sum", update_func="silu", aggr_update_func=None) for _ in range(num_layers)]
        )

        # Pre-pooling operation
        self.pre_pool = nn.ModuleDict({
            str(dim): 
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
            }
             for dim in range(self.max_dim+1)
        )

        # Post-pooling operation over all dimensions
        # and final classification
        self.post_pool = nn.Sequential(
            nn.Linear((max_dim + 1) * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )


    def forward(self, features: Dict[int, Tensor], edge_index_adjacencies: Dict[int, Tensor],
                edge_index_incidences: Dict[int, Tensor], invariances_r_r: Dict[int, Tensor],
                invariances_r_r_minus_1: Dict[int, Tensor]) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        features : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, channels)
            Input features on the cells of the simplicial complex.
        edge_index_incidences : dict[int, torch.sparse], length=max_rank, shape = (2, n_boundries_r_cells_r_cells)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        edge_index_adjacencies : dict[int, torch.sparse], length=max_rank, shape = (2, n_boundries_r_minus_1_cells_r_cells)
            Adjacency matrices :math:`H_r` mapping cells to cells via lower and upper cells.
        invariances_r_r : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_cells, n_rank_r_cells)
            Adjacency matrices :math:`I^0_r` with weights of cells to cells via lower and upper cells.
        invariances_r_r_minus_1 : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_minus_1_cells, n_rank_r_cells)
            Adjacency matrices :math:`I^1_r` with weights of map from r-cells to (r-1)-cells

        Returns
        -------
        Tensor, shape = (1)
            Regression value of the pooling of the graph
        """

        x = features 
        for layer in self.layers:
            x = layer(x, edge_index_adjacencies, edge_index_incidences, invariances_r_r, invariances_r_r_minus_1)

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


