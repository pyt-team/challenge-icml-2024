import torch
from torch_geometric.nn import GCNConv


class GCNModel(torch.nn.Module):
    def __init__(self, model_config, dataset_config):
        in_channels = (
            dataset_config["num_features"]
            if isinstance(dataset_config["num_features"], int)
            else dataset_config["num_features"][0]
        )
        hidden_channels = model_config["hidden_channels"]
        out_channels = dataset_config["num_classes"]
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv(x, edge_index)
        x = torch.relu(x)
        return self.linear(x)
