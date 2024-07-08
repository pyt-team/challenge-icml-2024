import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class NodeAttributeLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by grouping nodes with the same attribute.

    Parameters
    ----------
    attribute_idx : int
        The index of the node attribute to use for hyperedge construction.
    """

    def __init__(self, attribute_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.attribute_idx = attribute_idx

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain by grouping nodes with the same attribute.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        attribute = data.x[:, self.attribute_idx]
        unique_attributes = torch.unique(attribute)
        num_hyperedges = unique_attributes.size(0)
        # incidence matrix of the hypergraph
        incidence_1 = torch.zeros(data.num_nodes, num_hyperedges)
        for i, attr in enumerate(unique_attributes):
            nodes_with_attr = torch.where(attribute == attr)[0]
            incidence_1[nodes_with_attr, i] = 1

        incidence_1 = incidence_1.to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
