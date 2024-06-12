import torch
from topomodelx.nn.combinatorial.hmc import HMC


class HMCModel(torch.nn.Module):
    r"""HMC model that runs over combinatorial Complexes (CCs)

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
        negative_slope = model_config["negative_slope"]
        hidden_channels = model_config["hidden_channels"]
        out_channels = dataset_config["num_classes"]
        n_layers = model_config["n_layers"]
        super().__init__()
        channels_per_layer = []

        for l in range(n_layers):
            in_channels_l = []
            int_channels_l = []
            out_channels_l = []

            for i in range(3):
                # First layer behavior
                if l == 0:
                    in_channels_l.append(in_channels)
                else:
                    in_channels_l.append(hidden_channels)
                int_channels_l.append(hidden_channels)
                out_channels_l.append(hidden_channels)

            channels_per_layer.append((in_channels_l, int_channels_l, out_channels_l))

        self.base_model = HMC(
            channels_per_layer=channels_per_layer,
            negative_slope=negative_slope
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
        torch.Tensor
            Output tensor.
        """
        x_0 = data.x_0 
        x_1 = data.x_1
        x_2 = data.x_2
        neighborhood_0_to_0 = data['adjacency_0']
        neighborhood_1_to_1 = data['adjacency_1']
        neighborhood_2_to_2 = data['adjacency_2']
        neighborhood_0_to_1 = data['incidence_1']
        neighborhood_1_to_2 = data['incidence_2']

        x_0, x_1, x_2 = self.base_model(
            x_0,
            x_1,
            x_2,
            neighborhood_0_to_0,
            neighborhood_1_to_1,
            neighborhood_2_to_2,
            neighborhood_0_to_1,
            neighborhood_1_to_2,
        )

        x_0 = self.linear_0(x_0)
        x_1 = self.linear_1(x_1)
        x_2 = self.linear_2(x_2)
        return (x_0, x_1, x_2)
