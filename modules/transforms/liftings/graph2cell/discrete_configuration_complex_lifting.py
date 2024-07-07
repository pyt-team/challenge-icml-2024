import networkx as nx
import torch_geometric
from toponetx.classes import CellComplex

from modules.transforms.liftings.graph2cell.base import Graph2CellLifting


class CellCycleLifting(Graph2CellLifting):
    r"""Lifts graphs to cell complexes by generating the k-th *discrete configuration complex* $D_k(G)$ of the graph. This is a cube complex, which is similar to a simplicial complex except each n-dimensional cell is homeomorphic to a n-dimensional cube rather than an n-dimensional simplex.

    The discrete configuration complex of order k consists of all sets of k unique edges or vertices of $G$, with the additional constraint that if an edge e is in a cell, then neither of the endpoints of e are in the cell. For examples of different graphs and their configuration complexes, see the tutorial.

    Parameters
    ----------
    k : int
        The order of the configuration complex, or the number of 'points' in the configuration.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Generates the cubical complex of discrete graph configurations.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        G = self._generate_graph_from_data(data)
        cell_complex = CellComplex(G)
        return
