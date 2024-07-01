import networkx as nx
import torch

from networkx import Graph

from modules.transforms.liftings.graph2combinatorial.base import Graph2CombinatorialLifting

from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from topomodelx.utils.sparse import from_sparse

class NHopLifting(Graph2CombinatorialLifting):
    def __init__(self, hop_num: int = 1, **kwargs):
        self.hop_num = hop_num
        super().__init__(**kwargs)

    def lift_topology(self, data: Data) -> dict:

        # Create a netowrk X graph
        graph: Graph = to_networkx(data, to_undirected=True)
        new_graph = {}
        # Initialize the CCC with the base graph
        cc = CombinatorialComplex(graph, None, graph_based=True)

        hyper_edges = []

        # Get adjacency matrix from graph
        adj_mat = nx.adjacency_matrix(graph)
        previous_n_hop = {}

        # Iterate over the number of hops
        for _i in range(self.hop_num):
            # For each nodes neighbours
            for n in range(adj_mat.shape[0]):
                # Get the neighbours of the node and include the node itself
                k_hop_neighbours = set(adj_mat[n].nonzero()[1].tolist())

                # Add the node itself if empty
                if n not in previous_n_hop:
                    previous_n_hop[n] = set([n])

                # Make sure that n-hop neighbourhood is the neighbours at distance n
                # union with the (n-1)-neighbourhood
                k_hop_neighbours = k_hop_neighbours.union(previous_n_hop[n])

                hyper_edges.append(frozenset(k_hop_neighbours))

            # Multiply the adjacency matrix by itself to get the next hop
            adj_mat = adj_mat @ adj_mat

        new_graph["x_0"] = data["x"]

        # Create the incidence matrix for the hyperedges incidence [n_hyperedges, n_nodes]
        new_graph["incidence_hyperedges"] = torch.zeros(len(hyper_edges), graph.number_of_nodes())

        # Create the hyperedges so the n-th hyperedge has a 1 in the incidence matrix for
        # the nodes it connects
        for n, hyper_edge in enumerate(hyper_edges):
            new_graph["incidence_hyperedges"][n][list(hyper_edge)] = 1

        # Transpose so it fits conventions, [n_r_minus_1_cells, n_r_cells]
        new_graph["incidence_hyperedges"] = new_graph["incidence_hyperedges"].T.to_sparse_coo()

        # Create the incidence and adjacency matrices for the regular graph
        for r in range(cc.dim+1):
            if r < cc.dim:
                new_graph[f"incidence_{r+1}"] = from_sparse(cc.incidence_matrix(r, r+1, incidence_type="up"))
            new_graph[f"adjacency_{r}"] = from_sparse(cc.adjacency_matrix(r, r+1))
        return new_graph
