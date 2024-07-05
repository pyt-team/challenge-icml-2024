import math

import networkx as nx
import numpy as np
import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class HypergraphFormanRicciCurvatureLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain using Forman-Ricci curvature based backbone estimation.

    Parameters
    ----------
    network_type : str
        Network type may be weighted or unweighted. Default is "weighted".
    th_quantile: float
        Quantile to estimate cutoff threshold from Forman-Ricci curvature distribution to prune network and reveal backbone. Default is 0.6
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, network_type="weighted", th_quantile=0.6, **kwargs):
        super().__init__(**kwargs)
        self.network_type = network_type
        self.th_quantile = th_quantile

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain using Forman-Ricci curvature based backbone estimation.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """

        edge_list = data.edge_index.t().numpy()

        # for unweighted graphs or higher-dimensional edge or node features revert to unweighted network structure
        if (
            data.edge_attr is None
            or self.network_type == "unweighted"
            or data.edge_attr.shape[1] > 1
        ):
            edge_attr = np.ones(shape=(len(edge_list), 1))
        elif isinstance(data.edge_attr, torch.Tensor):
            edge_attr = data.edge_attr.numpy()
        else:
            edge_attr = data.edge_attr

        if data.x is None or self.network_type == "unweighted" or data.x.shape[1] > 1:
            node_attr = data.x = np.ones(shape=(data.num_nodes, 1))
        elif isinstance(data.x, torch.Tensor):
            node_attr = data.x.numpy()
        else:
            node_attr = data.x

        # create undirected networkx graph from pyg data
        G = nx.Graph()
        for v in range(len(node_attr)):
            G.add_node(v)
            G.nodes[v]["w"] = node_attr[v][0]

        for e in range(len(edge_list)):
            v1 = edge_list[e][0]
            v2 = edge_list[e][1]
            G.add_edge(v1, v2, w=edge_attr[e][0])
            G.add_edge(v1, v2, w=edge_attr[e][0])

        # estimate Forman-Ricci curvature as described in:
        # M. Weber, J. Jost, E. Saucan (2018). Detecting the Coarse Geometry of Networks. NeurIPS
        for v1, v2 in G.edges():
            v1_neighbors = set(G.neighbors(v1))
            v1_neighbors.remove(v2)
            v2_neighbors = set(G.neighbors(v2))
            v2_neighbors.remove(v1)

            w_e = G[v1][v2]["w"]
            w_v1 = G.nodes[v1]["w"]
            w_v2 = G.nodes[v2]["w"]
            ev1_sum = sum([w_v1 / math.sqrt(w_e * G[v1][v]["w"]) for v in v1_neighbors])
            ev2_sum = sum([w_v2 / math.sqrt(w_e * G[v2][v]["w"]) for v in v2_neighbors])

            G[v1][v2]["w_frc"] = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum))

        # estimate cutoff threshold from Forman-Ricci curvature distribution to prune network and reveal backbone(s), i.e. hyperedges
        w_frc = list(nx.get_edge_attributes(G, "w_frc").values())
        th_cutoff = np.quantile(w_frc, self.th_quantile)

        edges_to_remove = []
        for v1, v2 in G.edges():
            if G[v1][v2]["w_frc"] > th_cutoff:
                edges_to_remove.append((v1, v2))

        G.remove_edges_from(edges_to_remove)

        # find connected components (hyperedges)
        hyperedges = [
            c for c in sorted(nx.connected_components(G), key=len, reverse=True)
        ]
        shape = (data.num_nodes, len(hyperedges))
        incidence_matrix = np.zeros(shape=shape)

        if len(hyperedges) > 0:
            for i, nodes in enumerate(hyperedges):
                incidence_matrix[list(nodes), i] = 1

        incidences = torch.Tensor(incidence_matrix).to_sparse_coo()

        coo_indices = torch.stack(
            (incidences.indices()[0], incidences.indices()[1])  # nodes  # hyperedges
        )

        coo_values = incidences.values()

        incidence_matrix = torch.sparse_coo_tensor(coo_indices, coo_values)

        return {
            "incidence_hyperedges": incidence_matrix,
            "num_hyperedges": incidence_matrix.size(1),
            "x_0": data.x,
        }
