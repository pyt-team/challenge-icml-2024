from itertools import combinations

import networkx as nx
import numpy as np
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.convert import to_networkx

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


class Cy2CLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by using Cycle to Clique (Cy2C) algorithm based on Choi et al. (2022) (https://openreview.net/pdf?id=7d-g8KozkiE).

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_cycle_adj_speed_nosl(self, original_adj, data):
        Xgraph = to_networkx(data, to_undirected=True)
        num_g_cycle = (
            Xgraph.number_of_edges()
            - Xgraph.number_of_nodes()
            + nx.number_connected_components(Xgraph)
        )
        node_each_cycle = nx.cycle_basis(Xgraph)

        if num_g_cycle > 0:
            if len(node_each_cycle) != num_g_cycle:
                raise ValueError(
                    f"Number of cycles mismatch: local {len(node_each_cycle)}, total {num_g_cycle}"
                )

            cycle_adj = np.zeros(original_adj.shape)
            for nodes in node_each_cycle:
                for i in nodes:
                    cycle_adj[i, nodes] = 1
                cycle_adj[nodes, nodes] = 0
        else:
            node_each_cycle, cycle_adj = [], []

        return node_each_cycle, cycle_adj

    def _make_cy2c(
        self, data, max_node, cy2c_self=False, cy2c_same_attr=False, cy2c_trans=False
    ):
        v1, v2 = data.edge_index
        list_adj = torch.zeros((max_node, max_node))
        list_adj[v1, v2] = 1
        list_feature = data.x

        node_each_cycle, cycle_adj = self._make_cycle_adj_speed_nosl(list_adj, data)

        if len(cycle_adj) > 0:
            stacked_adjs = np.stack((list_adj, cycle_adj), axis=0)
        else:
            cycle_adj = np.zeros((1, list_adj.shape[0], list_adj.shape[1]))
            stacked_adjs = np.concatenate((list_adj[np.newaxis], cycle_adj), axis=0)

        edge_index = data.edge_index
        check_num = torch.sum(
            edge_index[0] - np.where(stacked_adjs[0] == 1)[0]
        ) + torch.sum(edge_index[1] - np.where(stacked_adjs[0] == 1)[1])
        if check_num != 0:
            print("error")
            return False

        cycle_index = torch.stack(
            (
                torch.LongTensor(np.where(stacked_adjs[1] != 0)[0]),
                torch.LongTensor(np.where(stacked_adjs[1] != 0)[1]),
            ),
            dim=0,
        )

        if cy2c_self:
            cycle_index, _ = remove_self_loops(
                cycle_index
            )  # Remove if self loops already exist
            cycle_index, _ = add_self_loops(cycle_index)
            cycle_attr = torch.ones(cycle_index.shape[1]).long()
        else:
            cycle_attr = torch.ones(cycle_index.shape[1]).long()

        if cy2c_same_attr:
            pos_edge_attr = torch.ones(edge_index.shape[1]).long()
        else:
            pos_edge_attr = torch.zeros(edge_index.shape[1]).long()

        if cy2c_trans:
            cycle_index, _ = remove_self_loops(cycle_index)
            old_length = cycle_index.shape[1]
            cycle_index, _ = add_self_loops(cycle_index)
            new_length = cycle_index.shape[1]
            cycle_attr = torch.ones(new_length).long()
            if new_length > old_length:
                cycle_attr[-(new_length - old_length) :] = 0

        return cycle_index, cycle_attr, pos_edge_attr, node_each_cycle

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
        max_node = data.num_nodes
        _, _, _, node_each_cycle = self._make_cy2c(data, max_node)

        simplices = [set() for _ in range(2, self.complex_dim + 1)]
        for nodes in node_each_cycle:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(nodes, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
