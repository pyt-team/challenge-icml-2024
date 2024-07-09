from itertools import combinations

import networkx as nx
import torch_geometric
from torch_geometric.utils.undirected import is_undirected
from toponetx.classes import SimplicialComplex


from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


class DirectedSimplicialCliqueLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying 
    the (k-1)-cliques as k-simplices if the clique has a single 
    source and sink.

    See [Computing persistent homology of directed flag complexes](https://arxiv.org/abs/1906.10458)
    for more details. 

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate_graph_from_data(self, data: torch_geometric.data.Data) -> nx.Graph:
        r"""Generates a NetworkX graph from the input data object. 
        Falls back to superclass method if data is not directed.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        nx.DiGraph
            The generated NetworkX graph.
        """
        # Check if undirected and fall back to superclass method if so
        if is_undirected(data.edge_index, data.edge_attr):
            return super()._generate_graph_from_data(data)

        # Check if data object have edge_attr, return list of tuples as [(node_id, {'features':data}, 'dim':1)] or ??
        nodes = [(n, dict(features=data.x[n], dim=0)) for n in range(data.x.shape[0])]

        if self.preserve_edge_attr and self._data_has_edge_attr(data):
            # In case edge features are given, assign features to every edge
            edge_index, edge_attr = (
                data.edge_index,
                data.edge_attr
            )
            edges = [
                (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
                for edge_idx, (i, j) in enumerate(
                    zip(edge_index[0], edge_index[1], strict=False)
                )
            ]
            self.contains_edge_attr = True
        else:
            # If edge_attr is not present, return list list of edges
            edges = [
                (i.item(), j.item())
                for i, j in zip(data.edge_index[0], data.edge_index[1], strict=False)
            ]
            self.contains_edge_attr = False
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex by identifying 
        the (k-1)-cliques of a graph as the k-simplices if the cliques have a single 
        source and sink.

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
        # find cliques in the undirected graph
        cliques = nx.find_cliques(graph.to_undirected())
        simplices = [set() for _ in range(2, self.complex_dim + 1)]
                    
        for clique in cliques:
            # locate the clique in the original directed graph
            gs = graph.subgraph(clique)
            # check if the clique has a single source and sink 
            # (i.e. is a DAG) and add as a simplex if so
            if nx.is_directed_acyclic_graph(gs):
                
                for i in range(2, self.complex_dim + 1):
                    for c in combinations(gs, i + 1):
                        simplices[i - 2].add(tuple(c))
            
        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
