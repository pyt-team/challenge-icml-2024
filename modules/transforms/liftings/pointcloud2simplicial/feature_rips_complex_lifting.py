import gudhi
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class FeatureRipsComplexLifting(PointCloud2SimplicialLifting):
    r"""Lifts point clouds to simplicial complex domain by generating the Vietoris-Rips complex using the Gudhi library. This complex is constructed in two steps - first add edges for all pairs of vertices a distance < d away from each other. Then generate the clique complex of the graph. Note that this implementation allows for *feature-based* distances as well - the distance of two nodes is a function of both the position and the features.

    If using feature-based distances, it is recommended that the features be normalized or standardized beforehand. This is because the Euclidean distance of the features will be used directly, so if the scales are wildy different the distances may be dominated by a small number of features with large magnitudes.

    Parameters
    ----------
    max_edge_length: float
        The maximum pairwise distance to add an edge to the graph

    feature_percent: float
        The percentage weight to give the feature-based distance (should be between 0 and 1)

    sparse: float or None.
        If float, uses a sparse approximation to the Rips complex to speed up computation.

    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_edge_length: float, feature_percent: float, **kwargs):
        self.feature_percent = feature_percent
        self.max_edge_length = max_edge_length
        self.sparse = self.sparse
        super().__init__(**kwargs)

    def generate_distance_matrix(self, data):
        """Generate the pairwise distance matrix of point cloud data, based on both point distances and feature-based distance.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch.tensor
            The pairwise distances.
        """
        x_expanded_1 = data.x.unsqueeze(1)
        x_expanded_2 = data.x.unsqueeze(0)

        # Calculate pairwise differences
        feature_differences = x_expanded_1 - x_expanded_2

        pass

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a point cloud to the Rips complex based on point-wise and feature-based distances.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        dm = self.generate_distance_matrix(data.pos, data.x)
        sc = gudhi.RipsComplex(
            distance_matrix=dm, sparse=self.sparse, max_edge_length=self.max_edge_length
        )
        stree = sc.create_simplex_tree()
        stree.prune_above_dimension(self.complex_dim)
        sc = SimplicialComplex(s for s, filtration_value in stree.get_simplices())
        lifted_topolgy = self._get_lifted_topology(sc)
        lifted_topolgy["x_0"] = data.x
        return lifted_topolgy
