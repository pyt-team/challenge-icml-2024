import torch
from topomodelx.nn.hypergraph.unigcn import UniGCN


class UniGCNModel(torch.nn.Module):
    r"""A simple UniGCN model that runs over hypergraph data.
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
        self.base_model = UniGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )
        self.linear_0 = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_hyperedges = torch.nn.Linear(hidden_channels, out_channels)

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
        x_0, x_hyperedges = self.base_model(data.x_0, data.incidence_hyperedges)
        x_0 = self.linear_0(x_0)
        x_hyperedges = self.linear_hyperedges(x_hyperedges)
        return (x_0, x_hyperedges)
