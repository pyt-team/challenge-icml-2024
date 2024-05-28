import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting
from modules.transforms.liftings.graph2simplicial.clique_lifting import (
    SimplicialCliqueLifting,
)


class SimplicialIndependentSetsLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the independent sets as k-simplices

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex by identifying the independent sets as k-simplices

        We use the fact that the independent sets of a graph G are the cliques of its complement graph Gc. The nodes and the edges
        of the complement graph represent the 0-simplices and 1-simplices respectively.

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
        complement_graph = nx.complement(graph)

        # Since we lose the original edges, not sure we can keep the edge features ? Should we prevent it ?
        self.contains_edge_attr = False

        # Propagate node features to complement
        nodes_attributes = {
            n: dict(features=data.x[n], dim=0) for n in range(data.x.shape[0])
        }
        nx.set_node_attributes(complement_graph, nodes_attributes)

        simplicial_complex = SimplicialComplex(complement_graph)
        simplices = SimplicialCliqueLifting.generate_simplices(
            self.complex_dim, complement_graph
        )

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
