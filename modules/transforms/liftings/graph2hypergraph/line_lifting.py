import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class HypergraphLineLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by considering line hypergraph.

    Line hypergraph is a hypergraph in which the vertices are the edges in the initial graph
    and a hyperedge connects the vertices in there's a vertex in the initial graph
    which is adjacent to the corresponding edges

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        # check for loops, since KNNGraph is inconsistent with nodes with equal features
        if self.loop:
            for i in range(num_nodes):
                if not torch.any(
                    torch.all(data_lifted.edge_index == torch.tensor([[i, i]]).T, dim=0)
                ):
                    connected_nodes = data_lifted.edge_index[
                        0, data_lifted.edge_index[1] == i
                    ]
                    dists = torch.sqrt(
                        torch.sum(
                            (data.pos[connected_nodes] - data.pos[i].unsqueeze(0) ** 2),
                            dim=1,
                        )
                    )
                    furthest = torch.argmax(dists)
                    idx = torch.where(
                        torch.all(
                            data_lifted.edge_index
                            == torch.tensor([[connected_nodes[furthest], i]]).T,
                            dim=0,
                        )
                    )[0]
                    data_lifted.edge_index[:, idx] = torch.tensor([[i, i]]).T

        incidence_1[data_lifted.edge_index[1], data_lifted.edge_index[0]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
