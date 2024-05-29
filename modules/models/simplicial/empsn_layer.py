import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor
from typing import Tuple, Dict, List, Literal
from modules.base.ScatterAggregation import ScatterAggregation
from modules.base.econv import EConv

class EMPSNLayer(torch.nn.Module):
    def __init__(
        self,
        channels,
        max_rank,
        n_inv: Dict[int, Dict[int, int]], # Number of invariances from r-cell to r-cell
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh", "silu"] | None = "sigmoid",
        aggr_update_func: Literal["relu", "sigmoid", "tanh", "silu"] | None = "sigmoid"
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_rank = max_rank
        
        # TODO Invariance dict
        # convolutions within the same rank
        self.convs_same_rank = torch.nn.ModuleDict(
            {
                f"rank_{rank}": 
                    EConv(
                        in_channels=channels,
                        weight_channels=n_inv[f"rank_{rank}"][f"rank_{rank}"], #from r-cell to r-cell
                        out_channels=1,
                        with_linear_transform_1=True,
                        with_linear_transform_2=True,
                        with_weighted_message=True,
                        update_func=update_func,
                    )

                for rank in range(max_rank) # Same rank conv up to r-1
            }
        )

        # convolutions from lower to higher rank
        self.convs_low_to_high = torch.nn.ModuleDict(
            {
                f"rank_{rank}": 
                    EConv(
                        in_channels=channels,
                        weight_channels=n_inv[f"rank_{rank-1}"][f"rank_{rank}"], #from r-1-cell to r-cell
                        out_channels=1,
                        with_linear_transform_1=True,
                        with_linear_transform_2=True,
                        with_weighted_message=True,
                        update_func=update_func,
                    )

                for rank in range(1, max_rank+1)
            }
        )

        # aggregation functions
        self.scatter_aggregations = torch.nn.ModuleDict(
            {
                f"rank_{rank}": ScatterAggregation(
                    aggr_func=aggr_func, 
                    update_func=aggr_update_func
                )
                for rank in range(max_rank + 1)
            }
        )
        # Perform an update over the received
        # messages by each other layer
        self.update = torch.nn.ModuleDict()
        for rank in range(max_rank+1):
            factor = 1
            for target_dict in n_inv.values():
                for target_rank in target_dict.keys():
                    factor += int(target_rank[-1] == str(rank))
            self.update[f"rank_{rank}"] = nn.Sequential(
                nn.Linear(factor*channels, channels),
                nn.SiLU(),
                nn.Linear(channels, channels)
                )
    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        for rank in self.convs_same_rank:
            self.convs_same_rank[rank].reset_parameters()
        for rank in self.convs_low_to_high:
            self.convs_low_to_high[rank].reset_parameters()

    def forward(self, features, adjacencies, incidences, invariances_r_r, invariances_r_r_minus_1) -> Dict[int, Tensor]:
        r"""Forward pass.

        Parameters
        ----------
        features : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, channels)
            Input features on the cells of the simplicial complex.
        edge_index_adjacencies : dict[int, torch.sparse], length=max_rank, shape = (2, n_boundries_r_minus_1_cells_r_cells)
            Adjacency matrices :math:`H_r` mapping cells to cells via lower and upper cells.
        edge_index_incidences : dict[int, torch.sparse], length=max_rank, shape = (2, n_boundries_r_cells_r_cells)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        invariances_r_r : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_cells, n_rank_r_cells)
            Adjacency matrices :math:`I^0_r` with weights of cells to cells via lower and upper cells.
        invariances_r_r_minus_1 : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_minus_1_cells, n_rank_r_cells)
            Adjacency matrices :math:`I^1_r` with weights of map from r-cells to (r-1)-cells

        Returns
        -------
        dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, channels)
            Output features on the cells of the simplicial complex.
        """

        for rank, feature in features.items():
            assert(not torch.isnan(feature).any())
            assert(not torch.isinf(feature).any())
        aggregation_dict = {} 

        h = {}

        # Same rank convolutions
        for rank in range(self.max_rank):
            # Get the convolution operation for the same rank
            conv = self.convs_same_rank[f"rank_{rank}"]

            rank_str = f"rank_{rank}"
            x_source = features[rank_str]
            x_target = features[rank_str]

            edge_index = adjacencies[rank_str]
            x_weights = invariances_r_r[rank_str]

            # print(rank, x_source.size(), x_target.size(), x_weights.size(), edge_index.size())

            send_idx, recv_idx = edge_index

            # Run the convolution
            message = conv(x_source, edge_index, x_weights, x_target)

            assert(message.size(0) == recv_idx.size(0))
            assert(not torch.isnan(message).any())
            assert(not torch.isinf(message).any())

            message_aggr = self.scatter_aggregations[rank_str](message, recv_idx, dim=0, target_dim=x_target.shape[0])

            aggregation_dict[rank_str] = {
                "message": [message_aggr],
                "recv_idx": recv_idx,
            }

        # Low to high convolutions starting from rank 1 and looking
        # backward to convolute
        for rank in range(1, self.max_rank+1):
            conv = self.convs_low_to_high[f"rank_{rank}"]

            rank_str = f"rank_{rank}"
            rank_minus_1_str = f"rank_{rank-1}"

            x_source = features[rank_minus_1_str]
            x_target = features[rank_str]
            x_weights = invariances_r_r_minus_1[rank_str]
            edge_index = incidences[rank_str]

            send_idx, recv_idx = edge_index

            # Run the convolution
            message = conv(x_source, edge_index, x_weights, x_target)
            message_aggr = self.scatter_aggregations[rank_str](message, recv_idx, dim=0, target_dim=x_target.shape[0])

            assert(message.size(0) == recv_idx.size(0))

            # Aggregate the message
            if rank_str not in aggregation_dict:
                aggregation_dict[rank_str] = {
                    "message": [message_aggr],
                    "recv_idx": recv_idx,
                }
            else:
                aggregation_dict[rank_str]['message'].append(message_aggr) 
                '''
                aggregation_dict[rank_str] = {
                    "message": [prev_msg, message],
                    #"recv_idx": torch.cat((aggregation_dict[rank]["recv_idx"], recv_idx)),
                    for prev_msg in aggregation_dict[rank_str]["message"]
                }
                '''

        for rank in range(self.max_rank+1):
            # Check for ranks not receiving any messages
            rank_str = f"rank_{rank}"
            if rank_str not in aggregation_dict:
                continue
            message = aggregation_dict[rank_str]["message"]
            recv_idx = aggregation_dict[rank_str]["recv_idx"]
            x_target = features[rank_str]

            # Aggregate the message
            #h[rank_str] = self.scatter_aggregations[rank_str](message, recv_idx, dim=0, target_dim=x_target.shape[0])

        # Update over the final embeddings with another MLP

        h = {}
        for rank, feature in features.items():
            feat_list = [feature]
            for msg_i in aggregation_dict[rank]["message"]:
                feat_list.append(msg_i)
            h[rank] = torch.cat(feat_list, dim=1)

        h = {
            rank: self.update[rank](feature)
            for rank, feature in h.items()
        }

        # Residual connection
        x = {rank: feature + h[rank] for rank, feature in features.items()}

        return x
