import torch
import torch_geometric
from torch_cluster import fps, radius

from modules.transforms.liftings.pointcloud2hypergraph.base import (
    PointCloud2HypergraphLifting,
)


class PointNetLifting(PointCloud2HypergraphLifting):
    r"""Lifts a point cloud to a hypergraph that is inspired by Qi et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets
    in a Metric Space" (NeurIPS 2027). This means building approximately equidistantly-spaced clusters of points via farthest point
    sampling and radius queries. Each of these clusters constitutes a hyperedge which is used for set abstraction.

    Parameters
    ----------
    sampling_ratio : float
        The sub-sampling ratio for farthest point sampling.
    cluster_radius : float
        The radius of the clusters w.r.t. the sub-sampled points.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, sampling_ratio: float, cluster_radius: float, **kwargs):
        super().__init__(**kwargs)

        self.sampling_ratio = sampling_ratio
        self.cluster_radius = cluster_radius

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to an expander hypergraph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """

        batch = (
            torch.zeros(data.num_nodes, dtype=torch.long)
            if data.batch is None
            else data.batch
        )

        sampling_idcs = fps(data.pos, batch, ratio=self.sampling_ratio)
        target_idcs, source_idcs = radius(
            data.pos,
            data.pos[sampling_idcs],
            self.cluster_radius,
            batch,
            batch[sampling_idcs],
        )

        incidence_matrix = torch.sparse_coo_tensor(
            torch.stack((source_idcs, target_idcs)),
            torch.ones(source_idcs.numel()),
            size=(data.num_nodes, sampling_idcs.numel()),
        )

        assert (
            1.0 not in incidence_matrix.sum(dim=1).to_dense()
        ), "Isolated cluster(s) detected, choose larger radius."

        return {
            "incidence_hyperedges": incidence_matrix,
            "num_hyperedges": incidence_matrix.size(1),
            "x_0": data.x,
        }
