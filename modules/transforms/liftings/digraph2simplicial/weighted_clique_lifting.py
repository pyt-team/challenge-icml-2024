from itertools import combinations
from math import e, isinf, sqrt

import networkx as nx
import numpy as np
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


class WeightedCliqueLifting(Graph2SimplicialLifting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        graph = self._generate_graph_from_data(data)

        graph = self.run(graph)

        graph = graph.to_undirected()

        simplicial_complex = SimplicialComplex(simplices=graph)

        cliques = nx.find_cliques(graph)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]

        for clique in cliques:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)

    def _generate_graph_from_data(self, data: torch_geometric.data.Data) -> nx.DiGraph:
        r"""Generates a NetworkX graph from the input data object.
        Overloaded for our purposes of using a directed graph
        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        nx.Graph
            The generated NetworkX graph.
        """
        G = torch_geometric.utils.to_networkx(data)
        G.clear_edges()

        for node in G.nodes():
            G.nodes[node]["features"] = data["features"][int(node)]

        for i in range(int(data["num_edges"])):
            G.add_edge(
                int(data["edge_index"][0][i]),
                int(data["edge_index"][1][i]),
                w=float(data["x"][i]),
            )

        return G

    # Summation for weighted FRC
    def summation(self, v_name, v_weight, e_weight, target, G):
        frc_sum = 0

        # since some nodes are undirected I will store seen nodes in a hashmap for instant lookup to see if it has already been traversed
        seen = set()

        # G.in_edges('node')'s formatting: tuple -> ('connecting_node', 'node')
        for v1, v2 in G.in_edges(target):
            if v_name is not v1 and v1 not in seen:
                frc_sum += v_weight / (sqrt(e_weight * G[v1][target]["w"]))
                seen.add(v1)

        # G.out_edges('node')'s formatting: tuple -> ('node', 'connecting_node')
        for v1, v2 in G.out_edges(target):
            if v_name is not v2 and v2 not in seen:
                frc_sum += v_weight / (sqrt(e_weight * G[target][v2]["w"]))
                seen.add(v2)

        return frc_sum

    # Weighted FRC builder -> return a hashmap where keys are a tuple of nodes and values are the edge's frc calculation
    def formanRicciCurvature(self, G, wHashmap):
        return_map = {}

        for v1, v2 in G.edges():
            # account for the weight of edges that aren't don't have incoming edges (they're not in the hashmap)
            target_edge_weight = G[v1][v2]["w"]
            if v1 in wHashmap:
                v1_weight = wHashmap[v1]
            else:
                v1_weight = 0

            if v2 in wHashmap:
                v2_weight = wHashmap[v2]
            else:
                v2_weight = 0

            # this formula has been taken from Jost Juergen's research papers
            frc = target_edge_weight * (
                (v1_weight / target_edge_weight)
                + (v2_weight / target_edge_weight)
                - self.summation(v2, v1_weight, target_edge_weight, v1, G)
                - self.summation(v1, v2_weight, target_edge_weight, v2, G)
            )

            # the caluclation throws a division by 0 so im using a try/except
            # the e-308 numbers is the smallest number that can fit in a float64

            try:
                distance_metric = 1 / (e**frc)
                if isinf(distance_metric):
                    distance_metric = 2.2250738585072014e-308
            except:
                distance_metric = 2.2250738585072014e-308

            if distance_metric < 2.2250738585072014e-308:
                distance_metric = 2.2250738585072014e-308

            if (v2, v1) not in return_map:
                return_map[(v1, v2)] = distance_metric
            else:
                return_map[(v2, v1)] = distance_metric

        return return_map

    # Normalization formula
    def formula(self, curr, max, min):
        return 1 / (1 + (9 * ((curr - min) / (max - min))))

    # Filtration function. Only keep edges that are below threshold.
    def removeEdges(self, graph_obj, threshold):
        graph_copy = graph_obj.copy()
        temp = set()

        for edge1, edge2 in graph_copy.edges():
            if graph_copy.has_edge(edge1, edge2):
                if graph_copy[edge1][edge2]["distance"] > (threshold):
                    temp.add((edge1, edge2))

        for e1, e2 in temp:
            graph_copy.remove_edge(e1, e2)

        return graph_copy

    def run(self, graph):
        weight_hashmap = {}
        for v1, v2 in graph.edges():
            if v2 not in weight_hashmap:
                weight_hashmap[v2] = graph[v1][v2]["w"]
            else:
                weight_hashmap[v2] = weight_hashmap[v2] + graph[v1][v2]["w"]

        temp_map = self.formanRicciCurvature(graph, weight_hashmap)

        # iterate through the hashmap -> each key is a pair of nodes
        # update the ['distance'] attribute to be the hashmap value
        # The try catch is incase the tuples got reversed.
        dist_arr = []
        for e1, e2 in graph.edges:
            try:
                graph[e1][e2]["distance"] = temp_map[(e1, e2)]
                dist_arr.append(temp_map[(e1, e2)])

            except:
                graph[e1][e2]["distance"] = temp_map[(e2, e1)]
                dist_arr.append(temp_map[(e2, e1)])

        # Filter the graph.
        G_copy = self.removeEdges(graph, np.percentile(dist_arr, 90))

        return G_copy
