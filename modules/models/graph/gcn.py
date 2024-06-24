import torch
from torch import Tensor
from torch_geometric.nn.models import GCN
from torch_geometric.utils import scatter


def global_mean_pool(x, batch=None, size=None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension.

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~torch_geometric.nn.aggr.MeanAggregation` module.

    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix :math:`\mathbf{X}`.
    batch : torch.Tensor, optional
        The batch vector :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`,
            which assigns each node to a specific example.
    size : int, optional
        The number of examples :math:`B`. Automatically calculated if not given.
    """
    dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2

    if batch is None:
        return x.mean(dim=dim, keepdim=x.dim() <= 2)
    return scatter(x, batch, dim=dim, dim_size=size, reduce="mean")


class GCNModel(torch.nn.Module):
    r"""A simple GCN model that runs over graph data.
    Note that some parameters are defined by the considered dataset.

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
        super().__init__()
        self.base_model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=n_layers,
        )
        self.pool = global_mean_pool

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
        z = self.base_model(data.x, data.edge_index)
        return self.pool(z)
