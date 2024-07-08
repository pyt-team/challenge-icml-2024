import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GraphSAGE


class GraphSAGEModel(torch.nn.Module):
    r"""A GraphSAGE model that performs graph classification.

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
        self.base_model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=n_layers,
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
        z = self.base_model(data.x, data.edge_index)
        return torch.nn.functional.softmax(global_mean_pool(z, None))
