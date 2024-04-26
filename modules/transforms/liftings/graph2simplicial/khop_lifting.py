import random
from itertools import combinations

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.io.utils.utils import get_complex_connectivity
from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


class SimplicialKHopLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by considering k-hop neighborhoods.

    Parameters
    ----------
    max_k_simplices : int, optional
        The maximum number of k-simplices to consider. Default is 5000.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_k_simplices=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_k_simplices = max_k_simplices

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to simplicial complex domain by considering k-hop neighborhoods.

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
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]
        for n in range(graph.number_of_nodes()):
            # Find 1-hop node n neighbors
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(n, 1, edge_index)
            if n not in neighbors:
                neighbors.append(n)
            neighbors = neighbors.numpy()
            neighbors = set(neighbors)
            for i in range(1, self.complex_dim):
                for c in combinations(neighbors, i + 1):
                    simplices[i - 1].add(tuple(c))
        for set_k_simplices in simplices:
            set_k_simplices = list(set_k_simplices)
            if len(set_k_simplices) > self.max_k_simplices:
                random.shuffle(set_k_simplices)
                set_k_simplices = set_k_simplices[: self.max_k_simplices]
            simplicial_complex.add_simplices_from(set_k_simplices)
        return self._get_lifted_topology(simplicial_complex, graph)
