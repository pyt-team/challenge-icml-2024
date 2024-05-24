from itertools import combinations

import networkx as nx
import torch_geometric
import gudhi
from gudhi import SimplexTree
from toponetx.classes import SimplicialComplex
import torch

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting, Graph2InvariantSimplicialLifting


def rips_lift(graph: torch_geometric.data.Data, dim: int, dis: float, fc_nodes: bool = True) -> SimplicialComplex:
    # create simplicial complex
    # Extract the node tensor and position tensor
    x_0, pos = graph.x, graph.pos

    # Create a list of each node tensor position 
    points = [pos[i].tolist() for i in range(pos.shape[0])]

    # Lift the graph to a Rips complex
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=dis)
    simplex_tree: SimplexTree  = rips_complex.create_simplex_tree(max_dimension=dim)

    # Add fully connected nodes to the simplex tree
    # (additionally connection between each pair of nodes u, v)

    if fc_nodes:
        nodes = [i for i in range(x_0.shape[0])]
        for edge in combinations(nodes, 2):
            simplex_tree.insert(edge)

    return SimplicialComplex.from_gudhi(simplex_tree)

class InvariantSimplicialVietorisRipsLifting(Graph2InvariantSimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the cliques as k-simplices.

    Parameters
    ----------
    distance : int, optional
        The distance for the Vietoris-Rips complex. Default is 0.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, dis: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.dis = dis
        self.contains_edge_attr = None

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


        # Lift graph to Simplicial Complex
        simplicial_complex = rips_lift(data, self.complex_dim, self.dis)

        # Retrieve features as a directory
        feature_dict = {}
        for i, node in enumerate(data.x):
            feature_dict[i] = node

        # Encode features in the simplex
        simplicial_complex.set_simplex_attributes(feature_dict, name='features')

        return self._get_lifted_topology(simplicial_complex, data)
class SimplicialVietorisRipsLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the cliques as k-simplices.

    Parameters
    ----------
    distance : int, optional
        The distance for the Vietoris-Rips complex. Default is 0.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, dis: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.dis = dis
        self.contains_edge_attr = None

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


        # Lift graph to Simplicial Complex
        simplicial_complex = rips_lift(data, self.complex_dim, self.dis)

        # Retrieve features as a directory
        feature_dict = {}
        for i, node in enumerate(data.x):
            feature_dict[i] = node

        # Encode features in the simplex
        simplicial_complex.set_simplex_attributes(feature_dict, name='features')

        return self._get_lifted_topology(simplicial_complex, data)