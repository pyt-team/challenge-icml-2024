import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.nn import global_add_pool
from typing import Tuple, Dict, List, Literal
from utils import compute_invariants_3d
from topomodelx.base.conv import Conv
from modules.base.econv import EConv
from topomodelx.base.aggregation import Aggregation

class EMPSNLayer(torch.nn.Module):
    def __init__(
        self,
        channels,
        max_rank,
        n_inv: Dict[int, Dict[int, int]], # Number of invariances from r-cell to r-cell
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh", "silu"] | None = "sigmoid"
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_rank = max_rank
        
        # TODO Invariance dict
        # convolutions within the same rank
        self.convs_same_rank = torch.nn.ModuleDict(
            {
                f"rank_{rank}": torch.nn.ModuleList(
                    [
                        EConv(
                        in_channels=2*channels + n_inv[rank][rank], #from r-cell to r-cell
                        out_channels=channels,
                        update_func=update_func,
                        ),

                        EConv(
                        in_channels=channels, 
                        out_channels=channels,
                        update_func=update_func,
                        ),
                        EConv(
                        in_channels=channels, 
                        out_channels=1,
                        update_func="sigmoid",
                        )
                    ]
                )
                for rank in range(max_rank) # Same rank conv up to r-1
            }
        )

        # convolutions from lower to higher rank
        self.convs_low_to_high = torch.nn.ModuleDict(
            {
                f"rank_{rank}": torch.nn.ModuleList(
                    [
                        EConv(
                            in_channels=2*channels + n_inv[rank-1][rank], # from r-1-cell to r-cell
                            out_channels=channels,
                            update_func=None,
                        ),
                        EConv(
                        in_channels=channels, 
                        out_channels=channels,
                        update_func="silu",
                        ),
                        EConv(
                        in_channels=channels, 
                        out_channels=1,
                        update_func="sigmoid",
                        )
                    ]
                )
                for rank in range(1, max_rank+1)
            }
        )

        # aggregation functions
        self.aggregations = torch.nn.ModuleDict(
            {
                f"rank_{rank}": Aggregation(
                    aggr_func=aggr_func, update_func=update_func
                )
                for rank in range(max_rank + 1)
            }
        )
        self.update = torch.nn.ModuleList(
            [
                EConv(
                    in_channels=channels,
                    out_channels=channels,
                    update_func=update_func,
                ),
                EConv(
                    in_channels=channels,
                    out_channels=channels,
                    update_func=None
                ),
            ]
            for rank in range(max_rank+1)
        )


    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        for rank in self.convs_same_rank:
            self.convs_same_rank[rank].reset_parameters()
        for rank in self.convs_low_to_high:
            self.convs_low_to_high[rank].reset_parameters()

    def forward(self, features, incidences, adjacencies, invariances_r_r, invariances_r_r_minus_1) -> Dict[int, Tensor]:
        r"""Forward pass.

        Parameters
        ----------
        features : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, channels)
            Input features on the cells of the simplicial complex.
        incidences : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_minus_1_cells, n_rank_r_cells)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        adjacencies : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_cells, n_rank_r_cells)
            Adjacency matrices :math:`H_r` mapping cells to cells via lower and upper cells.
        invariances_r_r : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_cells, n_rank_r_cells)
            Adjacency matrices :math:`I^0_r` with weights of cells to cells via lower and upper cells.
        invariances_r_r_minus_1 : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_minus_1_cells, n_rank_r_cells)
            Adjacency matrices :math:`I^1_r` with weights of map from r-cells to (r-1)-cells

        Returns
        -------
        dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, channels)
            Output features on the cells of the simplicial complex.
        """

        out_features = {}

        # Same rank convolutions
        for rank in range(self.max_rank):
            # Get the MLPs
            conv_msg_1 = self.convs_low_to_high[f"rank_{rank}"][0]
            conv_msg_2 = self.convs_low_to_high[f"rank_{rank}"][1]
            conv_edge_weights = self.convs_low_to_high[f"rank_{rank}"][2]

            x = features[rank]
            x_minus_1 = features[rank-1]
            n_r_cells = x.shape[0]
            adj = adjacencies[rank]
            inv_r = invariances_r_r[rank]
            inv_r_minus_1 = invariances_r_r_minus_1[rank]

            # For all r-cells reciving from r-cells
            for recv_idx in range(n_r_cells):
                # This are all nodes sending to for r-cell with index recv_idx 
                send_idx = adj.T[recv_idx] # This is a tensor with a one in the position of a neighboor
                state_send = x[send_idx]
                state_recv = x[recv_idx]
                edge_attr = inv_r.T[recv_idx] # Edge attributes of each receiving 

                # TODO is this correct ?
                current_state = torch.cat((state_send, state_recv, edge_attr), dim=1)

                message = conv_msg_2(conv_msg_1(current_state))
                edge_weights = conv_edge_weights(message)


                weighted_message = message * edge_weights

                weighted_message_aggr = self.aggregations[f"rank_{rank}"](weighted_message) 

                out_features[rank][recv_idx] = weighted_message_aggr

            # For all r-cells receiving from (r-1)-cells
            for recv_idx in range(n_r_cells):
                # This are all nodes sending to  r-cell with index recv_idx from (r-1)-cells
                send_idx = incidences.T[recv_idx] # This is a tensor with a one in the position of a neighboor
                state_send = x_minus_1[send_idx] # Recive the state from the (r-1)-cells
                state_recv = x[recv_idx] # The state of the r-cell
                edge_attr = inv_r_minus_1.T[recv_idx] # Edge attributes of each receiving  from (r-1)-cells to r-cells

                # TODO is this correct ?
                current_state = torch.cat((state_send, state_recv, edge_attr), dim=1)

                message = conv_msg_2(conv_msg_1(current_state))
                edge_weights = conv_edge_weights(message)


                weighted_message = message * edge_weights

                weighted_message_aggr = self.aggregations[f"rank_{rank}"](weighted_message) 

                out_features[rank][recv_idx] += weighted_message_aggr

        
        # MLP over the update
        h = {
            dim: self.update[dim][1](self.update[dim][0](feature)) # Run the 2 EConvs
            for dim, feature in out_features.items()
        }

        # Residual connection
        x = {dim: feature + h[dim] for dim, feature in x.items()}

        return x
