import torch
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

class ProjectionMean(torch_geometric.transforms.BaseTransform):
    r"""Lifts r-cell features to r+1-cells by projection using the mean

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
            if f"x_{elem}"  in data:
                continue
            idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1

            # r_cell (n_r_cells, channels)
            # incidence_r_r (n_(r-1)_cells, n_r_cells)
            # (r-1) cell (n_r-1_cells, channels)
            last_features = data[f"x_0"]

            for low_ele in range(1, idx_to_project+1):
                last_features = torch.matmul(
                    abs(data["incidence_" + low_ele].t()),
                    last_features,
                ) 

            # Take the mean
            last_features = last_features / (idx_to_project+1)
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