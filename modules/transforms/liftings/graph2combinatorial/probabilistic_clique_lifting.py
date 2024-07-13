from itertools import combinations

import networkx as nx
import torch_geometric
from toponetx.classes import CombinatorialComplex

from modules.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class ProbabilisticCliqueLifting(Graph2CombinatorialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the probabilistic cliques as k-cells.

    Parameters
    ----------
    probability : float
        A value from 0 to 1 indicating the probability that any given edge has been erased from the graph.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, probability=0.3, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a combinatorial complex by sequentially identifying the probabilistic cliques.
            A node is considered as part of a probabilistic clique if it is connected to >= (1-probability)*len(clique) many nodes in the clique.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.
        probability: float
            A value from 0 to 1 indicating the probability that any given edge has been erased from the graph.

        Returns
        -------
        dict
            The lifted topology.
        """

        if not (0 <= self.probability <= 1):
            raise ValueError("Probability must be between 0 and 1.")

        graph = self._generate_graph_from_data(data)
        combinatorial_complex = CombinatorialComplex(graph)

        clique_bins = self._clique_bins(graph)
        max_clique_size = max(clique_bins.keys())

        for dim in range(2, max_clique_size + 1):
            # Add neighbors to cliques of size dim
            for clique in clique_bins[dim]:
                neighbors = self._clique_neighbors(graph, clique)
                missing_edges = self._missing_edges(graph, clique, neighbors)
                graph.add_edges_from(missing_edges)

            # Re-identify cliques using the Bron-Kerbosch algorithm, which is more efficient than expanding existing cliques naively
            new_cliques = list(nx.find_cliques(graph))

            # Add cells of size dim+1
            for clique in new_cliques:
                for c in combinations(clique, dim + 1):
                    combinatorial_complex.add_cell(c, rank=dim)

        # Add remaining higher order cells
        new_max_clique_size = max(len(clique) for clique in new_cliques)
        for clique in new_cliques:
            for dim in range(max_clique_size + 1, new_max_clique_size):
                for c in combinations(clique, dim + 1):
                    combinatorial_complex.add_cell(c, rank=dim)

        lifted_topology = self._get_lifted_topology(combinatorial_complex)

        # Feature liftings
        lifted_topology["x_0"] = data.x
        return lifted_topology

    def _clique_bins(self, graph: nx.Graph):
        """
        Group cliques in the given graph by their size.

        Parameters
        ----------
        graph: nx.Graph

        Returns:
        dict: A dictionary where keys are clique sizes and values are lists of cliques of that size.
        """
        cliques = list(nx.find_cliques(graph))
        bins = {}

        for clique in cliques:
            size = len(clique)
            if size not in bins:
                bins[size] = []
            bins[size].append(tuple(clique))

        return bins

    def _clique_neighbors(self, graph: nx.Graph, clique: set) -> set:
        r"""
        Finds the nodes adjacent to the clique by at least (1-p)*len(clique) edges.

        Parameters
        ----------
        graph: nx.Graph
            The ambient graph
        clique: set
            A set of nodes that forms a clique in the graph.

        Returns
        -------
        set
            The set of node neighbors
        """
        threshold = (1 - self.probability) * len(clique)
        neighbors = set()

        for node in graph:
            if node not in clique:
                # Count the number of edges from this node to nodes in the clique
                common_neighbors = sum(
                    1 for neighbor in graph.neighbors(node) if neighbor in clique
                )
                # If the count meets the threshold, add the node to the neighbors set
                if common_neighbors >= threshold:
                    neighbors.add(node)

        return neighbors

    def _missing_edges(self, graph: nx.Graph, clique: set, nodes_to_connect: set):
        r"""
        Returns the missing edges from a set of nodes to a clique

        Parameters
        ----------
        graph: nx.Graph
            The ambient graph
        clique: set
        nodes_to_connect: set

        Returns
        -------
        list
            The missing edges
        """

        missing_edges = [
            (node, clique_node)
            for node in nodes_to_connect
            for clique_node in clique
            if not graph.has_edge(node, clique_node)
        ]

        return missing_edges
