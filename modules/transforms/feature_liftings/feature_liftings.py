import torch
import torch.nn.functional as F
import torch_geometric


class ProjectionSum(torch_geometric.transforms.BaseTransform):
    r"""Lifts r-cell features to r+1-cells by projection.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Projects r-cell features of a graph to r+1-cell structures using the incidence matrix.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data."""
        keys = sorted([key.split("_")[1] for key in data.keys() if "incidence" in key])  # noqa : SIM118
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                data["x_" + elem] = torch.matmul(
                    abs(data["incidence_" + elem].t()),
                    data[f"x_{idx_to_project}"],
                )
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Applies the lifting to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        return self.lift_features(data)

class ElementwiseMean(torch_geometric.transforms.BaseTransform):
    r"""Lifts r-cell features to r+1-cells by taking the mean of the lower
    dimensional features.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Projects r-cell features of a graph to r+1-cell structures using the incidence matrix.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data."""

        # Find the maximum dimension of the input data
        max_dim = max([int(key.split("_")[-1]) for key in data if "x_idx" in key])

        # Create a list of all x_idx tensors
        x_idx_tensors = [data[f"x_idx_{i}"] for i in range(max_dim + 1)]

        # Find the maximum sizes
        max_simplices = max(tensor.size(0) for tensor in x_idx_tensors)
        max_nodes = max(tensor.size(1) for tensor in x_idx_tensors)

        # Pad tensors to have the same size
        padded_tensors = [F.pad(tensor, (0, max_nodes - tensor.size(1), 0, max_simplices - tensor.size(0)))
                        for tensor in x_idx_tensors]

        # Stack all x_idx tensors
        all_indices = torch.stack(padded_tensors)

        # Create a mask for valid indices
        mask = all_indices != 0

        # Replace 0s with a valid index (e.g., 0) to avoid indexing errors
        all_indices = all_indices.clamp(min=0)

        # Get all embeddings at once
        all_embeddings = data["x_0"][all_indices]

        # Apply mask to set padded embeddings to 0
        all_embeddings = all_embeddings * mask.unsqueeze(-1).float()

        # Compute sum and count of non-zero elements
        embedding_sum = all_embeddings.sum(dim=2)
        count = mask.sum(dim=2).clamp(min=1)  # Avoid division by zero

        # Compute mean
        mean_embeddings = embedding_sum / count.unsqueeze(-1)

        # Assign results back to data dictionary
        for i in range(1, max_dim + 1):
            original_size = x_idx_tensors[i].size(0)
            data[f"x_{i}"] = mean_embeddings[i, :original_size]

        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Applies the lifting to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        return self.lift_features(data)
