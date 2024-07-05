import torch
from topomodelx.nn.combinatorial.hmc import HMC


class HMCModel(torch.nn.Module):
    r"""A simple CWN model that runs over combinatorial complex data.
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
        n_layers = model_config["n_layers"]
        hidden_channels = model_config["hidden_channels"]
        out_channels = dataset_config["num_classes"]

        channels_per_layer = [
            [
                [in_channels_0, in_channels_0, in_channels_0],
                [hidden_channels, hidden_channels, hidden_channels],
                [hidden_channels, hidden_channels, hidden_channels],
            ]
        ]
        rest = [
            [
                [hidden_channels for _ in range(3)],
                [hidden_channels for _ in range(3)],
                [hidden_channels for _ in range(3)],
            ]
            for __ in range(1, n_layers)
        ]
        channels_per_layer.extend(rest)

        negative_slope = model_config["negative_slope"]
        super().__init__()
        self.base_model = HMC(
            channels_per_layer=channels_per_layer, negative_slope=negative_slope
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
            data.adjacency_0,
            data.adjacency_1,
            data.adjacency_2,
            data.incidence_1,
            data.incidence_2,
        )
        x_0 = self.linear_0(x_0)
        x_1 = self.linear_1(x_1)
        x_2 = self.linear_2(x_2)
        return (x_0, x_1, x_2)
