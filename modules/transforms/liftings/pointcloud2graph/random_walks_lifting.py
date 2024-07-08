import random
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


class GraphRandomWalksLifting(PointCloud2GraphLifting):
    r"""Lifts point cloud data to graph by using Random walks.

    Parameters
    ----------
    k_value: int, optional
        The number of nearest neighbors to consider.
    **kwargs : optional
        Additional arguments for the class
    """

    def __init__(
        self, k: int = 5, num_walks: int = 100, steps_per_walk: int = 50, **kwargs
    ):
        super().__init__(**kwargs)
        self.k = k
        self.num_walks = num_walks
        self.steps_per_walk = steps_per_walk

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts a point cloud dataset to a graph by using Random walks.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted

        Returns
        -------
        dict
            The lifted topology containing the graph as torch_geometric.data.Data
        """
        point_cloud = data.pos.numpy()

        # Construct k-NN graph
        edge_index = knn_graph(torch.tensor(point_cloud), k=self.k)

        # Calculate distances and add them as edge weights
        edge_weights = self._calculate_edge_weights(point_cloud, edge_index)
        G = self._create_weighted_networkx_graph(edge_index, edge_weights)

        # Normalize edge weights to transition probabilities
        self._normalize_edge_weights(G)

        num_walks = self.num_walks
        steps_per_walk = self.steps_per_walk
        topological_graph = self._create_topological_graph(G, num_walks, steps_per_walk)

        # Convert the NetworkX graph back to PyTorch Geometric Data
        topological_edge_index = torch_geometric.utils.from_networkx(
            topological_graph
        ).edge_index
        lifted_data = Data(pos=data.pos, edge_index=topological_edge_index)

        return {
            "num_nodes": lifted_data.edge_index.unique().shape[0],
            "edge_index": lifted_data.edge_index,
        }

    def _calculate_edge_weights(self, point_cloud, edge_index):
        r"""Calculate the Euclidean distances for each edge.

        Parameters
        ----------
        point_cloud : np.ndarray
            The point cloud data
        edge_index : torch.Tensor
            The edge index tensor

        Returns
        -------
        np.ndarray
            The array of edge weights
        """
        weights = []
        for i, j in edge_index.t().tolist():
            weight = np.linalg.norm(point_cloud[i] - point_cloud[j])
            weights.append(weight)
        return np.array(weights)

    def _create_weighted_networkx_graph(self, edge_index, edge_weights):
        r"""Create a NetworkX graph with edge weights.

        Parameters
        ----------
        edge_index : torch.Tensor
            The edge index tensor
        edge_weights : np.ndarray
            The array of edge weights

        Returns
        -------
        networkx.DiGraph
            The NetworkX graph with edge weights
        """
        G = nx.DiGraph()
        for (i, j), weight in zip(edge_index.t().tolist(), edge_weights, strict=False):
            G.add_edge(i, j, weight=weight)
            G.add_edge(j, i, weight=weight)
        return G

    def _normalize_edge_weights(self, graph):
        r"""Normalize edge weights to transition probabilities.

        Parameters
        ----------
        graph : networkx.Graph
            The input graph with distance weights
        """
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            weights = [graph[node][neighbor]["weight"] for neighbor in neighbors]

            normalized_weights = torch.nn.functional.softmax(
                torch.tensor(weights)
            ).numpy()
            for i, neighbor in enumerate(neighbors):
                graph[node][neighbor]["weight"] = normalized_weights[i]

    def _random_walk(self, graph, start_node, steps):
        r"""Perform a random walk on the graph.

        Parameters
        ----------
        graph : networkx.Graph
            The input graph with transition probabilities
        start_node : int
            The starting node for the random walk
        steps : int
            The number of steps in the random walk

        Returns
        -------
        list
            The path of nodes visited during the random walk
        """
        current_node = start_node
        path = [current_node]

        for _ in range(steps):
            neighbors = list(graph.neighbors(current_node))
            probabilities = [
                graph[current_node][neighbor]["weight"] for neighbor in neighbors
            ]
            current_node = random.choices(neighbors, weights=probabilities, k=1)[0]
            path.append(current_node)

        return path

    def _create_topological_graph(self, graph, num_walks, steps_per_walk):
        r"""Create a topological graph based on random walks.

        Parameters
        ----------
        graph : networkx.Graph
            The input graph with transition probabilities
        num_walks : int
            The number of random walks to perform from each node
        steps_per_walk : int
            The number of steps per random walk

        Returns
        -------
        networkx.Graph
            The constructed topological graph
        """
        transition_counts = defaultdict(int)
        num_nodes = graph.number_of_nodes()

        # Perform multiple random walks
        for start_node in range(num_nodes):
            for _ in range(num_walks):
                path = self._random_walk(graph, start_node, steps_per_walk)
                for i in range(len(path) - 1):
                    transition_counts[(path[i], path[i + 1])] += 1
                    transition_counts[(path[i + 1], path[i])] += 1

        # Create new graph based on transition counts
        H = nx.Graph()
        for (u, v), count in transition_counts.items():
            if count > 0:  # Threshold to filter out insignificant transitions
                H.add_edge(u, v, weight=count)

        return H
