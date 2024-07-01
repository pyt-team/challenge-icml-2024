import numpy as np
import potpourri3d as pp3d
import scipy
import torch
import torch_geometric
from tqdm.auto import tqdm

from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


class GeodesicLifting(PointCloud2GraphLifting):
    r"""Lifts pointcloud to geodesic graph.
    Parameters
    ----------
    distance : float
        The desired geodesic distance for which the graph should be constructed.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, distance: float, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts pointcloud to geodesic graph.
        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.
        Returns
        -------
        dict
            The lifted topology.
        """

        # Solve all-pair geodesic with heat method
        solver = pp3d.PointCloudHeatSolver(data.pos.numpy())
        geodesics = np.stack(
            [solver.compute_distance(i) for i in tqdm(range(data.pos.size(0)))],
            0,
        )

        # Assemble hypergraph
        incidence_matrix = scipy.sparse.coo_matrix(
            (geodesics <= self.distance) * geodesics
        )
        coo_indices = torch.stack(
            (
                torch.from_numpy(incidence_matrix.row),
                torch.from_numpy(incidence_matrix.col),
            )
        )
        coo_values = torch.from_numpy(incidence_matrix.data.astype("f4"))
        incidence_matrix = torch.sparse_coo_tensor(coo_indices, coo_values)

        return {
            "edges": incidence_matrix,
            "num_edges": coo_values.size(0),
            "x_0": data.x,
        }
