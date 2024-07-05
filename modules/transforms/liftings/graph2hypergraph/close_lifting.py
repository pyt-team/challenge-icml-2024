import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class HypergraphCloseLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by considering k-nearest neighbors.

    Parameters
    ----------
    k_value : int, optional
        The number of nearest neighbors to consider. Default is 1.
    loop: boolean, optional
        If True the hyperedges will contain the node they were created from.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, distance, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance

    def find_close_res(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Finds the closest nodes to each node in the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The data with the closest nodes.
        """
        # In data there is the nodes and the distances between them
        # We need to find the closest nodes to each node
        distances = data.edge_attr[0]
        # distances is a list of distances between nodes
        # the nodes are specified in data.edge_index
        # we need to find the nodes closer than a certain distance
        # to each node
        # want to create a list with the close nodes to each node
        # For instance, if the distance is 3, and the nodes are 0, 1, 2, 3, 4, 5
        # and the distances are [1, 2, 3, 4, 5, 6] of the first node vs all
        # the other nodes, the closest nodes are 0, 1, 2

        # Want to return [[0, 1, 2], ...]
        # where the first list is the index of the closest nodes to the first node
        # the second list is the index of the closest nodes to the second node
        num_nodes = data.x.shape[0]
        closest_nodes = []
        for i in range(num_nodes):
            # Get the distances of the ith node
            distances_i = distances[data.edge_index[0] == i]
            # Get the indices of the closest nodes
            closest_nodes_i = torch.where(distances_i < self.distance)[0]
            closest_nodes.append(closest_nodes_i)
        return closest_nodes

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain by considering k-nearest neighbors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        num_nodes = data.x.shape[0]
        data.pos = data.x
        # data_lifted = self.transform(data)

        # Find the closest nodes to each node
        closest_nodes = self.find_close_res(data)

        # Now, I want to create hyperedges of the closest nodes
        # Hyperedges = closest_nodes + edges
        num_hyperedges = len(closest_nodes) + len(data.edge_index[0])

        incidence_1 = torch.zeros(num_nodes, num_hyperedges)
        incidence_1[data.edge_index[1], data.edge_index[0]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
