from itertools import combinations

import networkx as nx
import torch_geometric
import gudhi
from gudhi import SimplexTree
from toponetx.classes import SimplicialComplex
import torch

from modules.transforms.liftings.pointcloud2simplicial.base import PointCloud2SimplicialLifting

class SimplicialAlphaComplexLifting(PointCloud2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the cliques as k-simplices.

    Parameters
    ----------
    distance : int, optional
        The distance for the Vietoris-Rips complex. Default is 0.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex by identifying the cliques as k-simplices.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """


        pos = data.pos

        # Create a list of each node tensor position 
        points = [pos[i].tolist() for i in range(pos.shape[0])]

        # Lift the graph to an AlphaComplex
        alpha_complex = gudhi.AlphaComplex(points=points)
        simplex_tree: SimplexTree = alpha_complex.create_simplex_tree()
        simplicial_complex = SimplicialComplex.from_gudhi(simplex_tree)

        feature_dict = {

        }

        for i, node in enumerate(data.x):
            feature_dict[i] = node

        simplicial_complex.set_simplex_attributes(feature_dict, name='features')

        # Assign feature embeddings to the SimplicialComplex for 0-simplices (nodes)
        # and then for higher order n-simplices by taking the mean of the lower order simplices

        # TODO Add edge_attributes 

        return self._get_lifted_topology(simplicial_complex, data)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        r"""Applies the full lifting (topology + features) to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        return torch_geometric.data.Data(**initial_data, **lifted_topology)