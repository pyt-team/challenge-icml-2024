from itertools import combinations

import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


class SimplicialCliqueLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the cliques as k-simplices.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def generate_simplices(dim: int, graph: nx.Graph) -> list:
        r"""Generates list of simplices from cliques of a given graph

        Parameters
        ----------
        dim: int
            Maximum dimension of the complex
        graph: nx.Graph
            Input graph

        Returns
        -------
        list[tuple]
            List of simplices
        """
        cliques = nx.find_cliques(graph)
        simplices = [set() for _ in range(2, dim + 1)]
        for clique in cliques:
            for i in range(2, dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))
        return simplices

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
        graph = self._generate_graph_from_data(data)
        simplicial_complex = SimplicialComplex(graph)

        simplices = self.generate_simplices(self.complex_dim, graph)
        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
