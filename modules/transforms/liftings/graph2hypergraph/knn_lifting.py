import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class HypergraphKNNLifting(Graph2HypergraphLifting):
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

    def __init__(self, k_value=1, loop=True, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value
        self.loop = loop
        self.transform = torch_geometric.transforms.KNNGraph(self.k, self.loop)

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
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_nodes)
        data_lifted = self.transform(data)
        incidence_1[data_lifted.edge_index[1], data_lifted.edge_index[0]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
