import torch
from topomodelx.nn.combinatorial.hmc import HMC


class HMCModel(torch.nn.Module):
    r"""A simple HMC model that runs over combinatorial complex data.
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
        negative_slope = model_config["negative_slope"]

        super().__init__()

        in_channels_layer = [in_channels, in_channels, in_channels]
        int_channels_layer = [hidden_channels, hidden_channels, hidden_channels]
        out_channels_layer = [hidden_channels, hidden_channels, hidden_channels]

        channels_per_layer = [
            [in_channels_layer, int_channels_layer, out_channels_layer]
        ]

        for _ in range(1, n_layers):
            in_channels_layer = [hidden_channels, hidden_channels, hidden_channels]
            int_channels_layer = [hidden_channels, hidden_channels, hidden_channels]
            out_channels_layer = [hidden_channels, hidden_channels, hidden_channels]

            channels_per_layer.append(
                [in_channels_layer, int_channels_layer, out_channels_layer]
            )

        self.base_model = HMC(
            channels_per_layer=channels_per_layer, negative_slope=negative_slope
        )
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

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
        x = self.base_model(
            data.x_0,
            data.x_1,
            data.x_2,
            data.adjacency_0,
            data.adjacency_1,
            data.adjacency_2,
            data.incidence_1,
            data.incidence_2,
        )[1]

        x = self.linear(x)

        return torch.sigmoid(x)
