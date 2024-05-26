from typing import Literal

import torch
from torch_scatter import scatter_add, scatter_mean


class ScatterAggregation(torch.nn.Module):
    """Message passing layer.

    Parameters
    ----------
    aggr_func : {"mean", "sum"}, default="sum"
        Aggregation method (Inter-neighborhood).
    update_func : {"relu", "sigmoid", "tanh", None}, default="sigmoid"
        Update method to apply to merged message.
    """

    def __init__(
        self,
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh"] | None = "sigmoid",
    ) -> None:
        super().__init__()
        self.aggr_func = aggr_func
        self.update_func = update_func

    def update(self, inputs):
        """Update (Step 4).

        Parameters
        ----------
        input : torch.Tensor
            Features for the update step.

        Returns
        -------
        torch.Tensor
            Updated features with the same shape as input.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(inputs)
        if self.update_func == "relu":
            return torch.nn.functional.relu(inputs)
        if self.update_func == "tanh":
            return torch.tanh(inputs)
        return None

    def forward(self, x, index: torch.Tensor, dim: int = 0, target_dim: int = 0):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            A Tensor containing the messages that should be merged  the dimension
        index : torch.Tensor
            An Tensor containing indices of the messages to aggregate.
        dim : int 
            The dimension of the `x` tensor in which to aggregate the results

        Returns
        -------
        torch.Tensor
            Aggregated messages.
        """
        if self.aggr_func == "sum":
            x = scatter_add(x, index, dim=dim, dim_size=target_dim)
        if self.aggr_func == "mean":
            x = scatter_mean(x, index, dim=dim)
        if self.update_func is not None:
            x = self.update(x)
        return x
