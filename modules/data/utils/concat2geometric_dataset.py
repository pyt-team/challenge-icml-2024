from torch_geometric.data import Data, Dataset


class ConcatToGeometricDataset(Dataset):
    r"""Concatenates list of PyTorch Geometric Data objects to form a dataset."""

    def __init__(self, concat_dataset):
        r"""Initializes the dataset.

        Parameters
        ----------
        concat_dataset : list
            List of PyTorch Geometric Data objects to be concatenated.
        """
        super().__init__()
        self.concat_dataset = concat_dataset

    def len(self):
        r"""Returns the length of the dataset."""
        return len(self.concat_dataset)

    def get(self, idx):
        r"""Returns the PyTorch Geometric Data object at the specified index.

        Parameters
        ----------
        idx : int
            Index of the data object to be returned.
        """
        data = self.concat_dataset[idx]

        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        y = data.y
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=1)
        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr.unsqueeze(dim=1)
        if len(y.shape) == 1:
            y = y.unsqueeze(dim=1)

        # Construct PyTorch Geometric Data object
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
