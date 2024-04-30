import torch
from topomodelx.nn.cell.cwn import CWN


class CWNModel(torch.nn.Module):
    r"""A simple CWN model that runs over cell complex data.
    Note that some parameters are defined by the considered dataset.

    Parameters
    ----------
    model_config : Dict | DictConfig
        Model configuration.
    dataset_config : Dict | DictConfig
        Dataset configuration.
    """

    def __init__(self, model_config, dataset_config):
        in_channels_0 = (
            dataset_config["num_features"]
            if isinstance(dataset_config["num_features"], int)
            else dataset_config["num_features"][0]
        )
        in_channels_1 = in_channels_0
        in_channels_2 = in_channels_0
        hidden_channels = model_config["hidden_channels"]
        out_channels = dataset_config["num_classes"]
        n_layers = model_config["n_layers"]
        super().__init__()
        self.base_model = CWN(
            in_channels_0,
            in_channels_1,
            in_channels_2,
            hidden_channels,
            n_layers,
        )
        self.linear_0 = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_1 = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        r"""Forward pass of the model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data.

        Returns
        -------
        tuple of torch.Tensor
            Output tensor.
        """
        x_0, x_1, x_2 = self.base_model(
            data.x_0,
            data.x_1,
            data.x_2,
            data.adjacency_1,
            data.incidence_2,
            data.incidence_1.T,
        )
        x_0 = self.linear_0(x_0)
        x_1 = self.linear_1(x_1)
        x_2 = self.linear_2(x_2)
        return (x_0, x_1, x_2)
