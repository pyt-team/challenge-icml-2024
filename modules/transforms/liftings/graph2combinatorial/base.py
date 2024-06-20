import networkx as nx
import torch
from toponetx.classes import CombinatorialComplex

from modules.data.utils.utils import get_combinatorial_connectivity
from modules.transforms.liftings.lifting import GraphLifting


class Graph2CombinatorialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2combinatorial"

    def _get_lifted_topology(self, combinatorial_complex: CombinatorialComplex, graph: nx.Graph) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        combinatorial_complex : CellComplex
            The cell complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_combinatorial_connectivity(combinatorial_complex, self.complex_dim)
        # lifted_topology["x_0"] = torch.stack(
        #     list(combinatorial_complex.get_cell_attributes("features", 0).values())
        # )
        # # If new edges have been added during the lifting process, we discard the edge attributes
        # if self.contains_edge_attr and combinatorial_complex.shape[1] == (
        #     graph.number_of_edges()
        # ):
        #     lifted_topology["x_1"] = torch.stack(
        #         list(combinatorial_complex.get_cell_attributes("features", 1).values())
        #     )
        return lifted_topology
