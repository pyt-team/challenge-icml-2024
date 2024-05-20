"""Convolutional layer for message passing."""
import math
from typing import Literal

import torch
from torch.nn.parameter import Parameter
from torch_scatter import scatter_add

from topomodelx.base.conv import Conv


class EConv(Conv):
    """Message passing: steps 1, 2, and 3.

    Builds on the Conv class to implement message passing but 
    adds MLPs to different sections of the convolution procedure
    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    weight_channels : int
        Dimension of the edge weights.
    out_channels : int
        Dimension of output features.
    aggr_norm : bool, default=False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : {"relu", "sigmoid", "silu"}, optional
        Update method to apply to message.
    att : bool, default=False
        Whether to use attention.
    initialization : {"xavier_uniform", "xavier_normal"}, default="xavier_uniform"
        Initialization method.
    initialization_gain : float, default=1.414
        Initialization gain.
    with_linear_transform_1 : bool, default=True
        Whether to apply a first learnable linear transform.
        NB: if `False` in_channels has to be equal to out_channels.
    with_linear_transform_2 : bool, default=True
        Whether to apply a second learnable linear transform.
    with_weighted_message : bool, default=True
        Whether to apply a learnable message weighting
    """

    def __init__(
        self,
        in_channels,
        weight_channels,
        out_channels,
        aggr_norm: bool = False,
        update_func: Literal["relu", "sigmoid", "silu", None] = None,
        att: bool = False,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        initialization_gain: float = 1.414,
        with_linear_transform_1: bool = True,
        with_linear_transform_2: bool = True,
        with_weighted_message: bool = True
    ) -> None:
        super().__init__(
            att=att,
            initialization=initialization,
            initialization_gain=initialization_gain,
        )
        self.weight_channels = weight_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight_1 = (
            Parameter(torch.Tensor(2*self.in_channels + weight_channels, self.in_channels))
            if with_linear_transform_1
            else None
        )
        self.weight_2 = (
            Parameter(torch.Tensor(self.in_channels, self.in_channels))
            if with_linear_transform_2
            else None
        )

        self.weight_3 = (
            Parameter(torch.Tensor(self.in_channels, self.out_channels))
            if with_weighted_message
            else None
        )

        # TODO correct different cases here
        if not with_linear_transform_1 and not with_linear_transform_2 and not with_weighted_message and in_channels != out_channels:
            raise ValueError(
                "With `linear_trainsform=False`, in_channels has to be equal to out_channels"
            )

        # TODO correct attention possibility
        if self.att:
            self.att_weight = Parameter(
                torch.Tensor(
                    2 * self.in_channels,
                )
            )

        self.reset_parameters()

    def reset_parameters(self):
        match self.initialization:
            case "uniform":
                if self.weight_1 is not None:
                    stdv = 1.0 / math.sqrt(self.weight.size(1))
                    self.weight_1.data.uniform_(-stdv, stdv)
                if self.weight_2 is not None:
                    stdv = 1.0 / math.sqrt(self.weight.size(1))
                    self.weight_1.data.uniform_(-stdv, stdv)
                if self.weight_2 is not None:
                    stdv = 1.0 / math.sqrt(self.weight.size(1))
                    self.weight_1.data.uniform_(-stdv, stdv)
                if self.att:
                    stdv = 1.0 / math.sqrt(self.att_weight.size(1))
                    self.att_weight.data.uniform_(-stdv, stdv)
            case "xavier_uniform":
                if self.weight_1 is not None:
                    torch.nn.init.xavier_uniform_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.weight_2 is not None:
                    torch.nn.init.xavier_uniform_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.weight_3 is not None:
                    torch.nn.init.xavier_uniform_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_uniform_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case "xavier_normal":
                if self.weight_1 is not None:
                    torch.nn.init.xavier_normal_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.weight_2 is not None:
                    torch.nn.init.xavier_normal_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.weight_3 is not None:
                    torch.nn.init.xavier_normal_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_normal_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case _:
                raise ValueError(
                    f"Initialization {self.initialization} not recognized."
                )


    def update(self, x_message_on_target) -> torch.Tensor:
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape = (n_target_cells, out_channels)
            Output features on target cells.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)
        if self.update_func == 'silu':
            return torch.nn.functional.silu(x_message_on_target)
        return x_message_on_target

    def forward(self, x_source, edge_index, x_weights, x_target=None) -> torch.Tensor:
        """Forward pass.

        This implements message passing:
        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells with input features `x_target`.
        - using a certain set of weights `x_weights`

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_weights : Tensor, shape = (..., n_edges, weight_channels)
            Weights of the relation connecting `x_source` and `x_target`.
        neighborhood : torch.sparse, shape = (2, n_edges)
            Edge index format for neighborhood.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_target_cells, out_channels)
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """

        # TODO Fix attention
        '''
        if self.att:
            neighborhood = neighborhood.coalesce()
            self.target_index_i, self.source_index_j = neighborhood.indices()
            attention_values = self.attention(x_source, x_target)
            neighborhood = torch.sparse_coo_tensor(
                indices=neighborhood.indices(),
                values=attention_values * neighborhood.values(),
                size=neighborhood.shape,
            )
        '''
        # Construct the edge index tensor of size (2, n_boundaries)
        send_idx, recv_idx = edge_index

        x_message = torch.cat((x_source[send_idx], x_target[recv_idx], x_weights), dim=1) 

        if self.weight_1 is not None:
            x_message = torch.mm(x_message, self.weight_1)
            x_message = self.update(x_message)
        if self.weight_2 is not None:
            x_message = torch.mm(x_message, self.weight_2)
            x_message = self.update(x_message)
        if self.weight_3 is not None:
            x_message_weights =  torch.mm(x_message, self.weight_3)
            self.x_message_weights = torch.nn.Sigmoid(x_message_weights)
        else:
            x_message_weights = torch.ones_like(x_message)

        # Weight the message by the learned weights
        x_message = x_message * x_message_weights

        # TODO correct aggregation
        '''
        if self.aggr_norm:
            neighborhood_size = torch.sum(neighborhood.to_dense(), dim=1)
            x_message_on_target = torch.einsum(
                "i,ij->ij", 1 / neighborhood_size, x_message_on_target
            )
        '''

        return scatter_add(x_message, recv_idx, dim=0, dim_size=x_target.size(0))