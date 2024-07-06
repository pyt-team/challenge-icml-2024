import torch
import torch_geometric
from torch_cluster import fps, knn

from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


class VoronoiLifting(PointCloud2GraphLifting):
    r"""Lifts pointcloud to voronoi graph.

    Parameters
    ----------
    support_ratio : float
        Ratio of points to sample with FPS to form voronoi support set.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, support_ratio: float, **kwargs):
        super().__init__(**kwargs)
        self.support_ratio = support_ratio

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts pointcloud to voronoi graph induced by FPS support set.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        support_indices = fps(data.pos, ratio=self.support_ratio)
        pool_target, pool_source = knn(data.pos[support_indices], data.pos, k=1)
        edges = torch.stack([pool_target, pool_source])

        return {
            "edge_index": edges,
            "x_0": data.x,
        }
