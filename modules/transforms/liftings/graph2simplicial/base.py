import networkx as nx
import torch
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.lifting import GraphLifting


class Graph2SimplicialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "graph2simplicial"
        self.signed = kwargs.get("signed", False)

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex, graph: nx.Graph
    ) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """

        lifted_topology = get_complex_connectivity(
            simplicial_complex, self.complex_dim, signed=self.signed
        )

        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )

        # If new edges have been added during the lifting process, we discard the edge attributes
        if self.contains_edge_attr and simplicial_complex.shape[1] == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                list(simplicial_complex.get_simplex_attributes("features", 1).values())
            )

        return lifted_topology
