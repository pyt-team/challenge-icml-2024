import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool
from modules.models.simplicial.empsn_layer import EMPSNLayer
from typing import Tuple, Dict, List, Literal
from modules.models.model_utils import decompose_batch


class EMPSNModel(nn.Module):
    r"""
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)

    Parameters
    ----------
    model_config : Dict | DictConfig
        Model configuration.
    dataset_config : Dict | DictConfig
        Dataset configuration.
    """

    def __init__(self, model_config, dataset_config):
        in_channels = (
            dataset_config["num_features"]
            if isinstance(dataset_config["num_features"], int)
            else dataset_config["num_features"][0]
        )
        hidden_channels = model_config["hidden_channels"]
        out_channels = dataset_config["num_classes"]
        n_layers = model_config["n_layers"]
        max_dim = model_config["max_dim"]
        inv_dims = model_config["inv_dims"]

        super().__init__()
        self.base_model = EMPSN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            max_dim=max_dim,
            n_layers=n_layers,
            inv_dims=inv_dims
        )
    def forward(self, data):
        r"""Forward pass of the model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.base_model(data)
        return x

class EMPSN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int, max_dim: int, inv_dims: Dict) -> None: 
        super().__init__()

        self.max_dim = max_dim
        self.feature_embedding = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [EMPSNLayer(hidden_channels, self.max_dim, aggr_func="sum", update_func="silu", aggr_update_func=None, n_inv=inv_dims) for _ in range(n_layers)]
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


