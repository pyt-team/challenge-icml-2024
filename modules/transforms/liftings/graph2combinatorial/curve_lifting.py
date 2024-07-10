from collections.abc import Collection

import torch_geometric
import torch_geometric.data

from modules.matroid.matroid import CCMatroid, powerset
from modules.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
    Graph2MatroidLifting,
    Matroid2CombinatorialLifting,
)


class GraphCurveMatroidLifting(Graph2MatroidLifting):
    r"""Lifts graphs to graph curve matroids by identifying cycles or certain acycles as n-cells.

    Parameters
    ----------
    max_rank : int, optional
        The maximum length of the complex to be lifted. Default is None.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_rank: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.max_rank = max_rank

    def edges_incident(
        self, v: Collection | int, data: torch_geometric.data.Data
    ) -> list[frozenset]:
        """Returns edges incident to v based on (graph) data

        Parameters
        ----------
        v: Collection | int
            vertex or list of vertices to find incidence
        data : torch_geometric.data.Data
            pytorch geometric data that contains data of a graph.

        Returns
        -------
        list[frozenset]
            The list of undirected edges incident to v.
        """
        return [
            frozenset(edge)
            for edge in self.get_edges_incident(vertex=v, data=data).t().tolist()
        ]

    def rank_check(
        self, dual: CCMatroid, V: Collection, data: torch_geometric.data.Data
    ) -> set[frozenset]:
        """Part of the Algorithm proposed by Geiger et. al
        We compute the graph curve matroid by computing its circuits.
        This function serves as a preprocessing for calculating the circuits.
        Taken from https://mathrepo.mis.mpg.de/_downloads/30a7910c728d51fc01271eb30e46a42a/graphCurveMatroid.m2

        Parameters
        ----------
        dual: CCMatroid
            the dual matroid of a given graphic matroid connected to data.
        V: Collection | int
            vertex or set of vertices to find incidence
        data : torch_geometric.data.Data
            pytorch geometric data that contains data of a graph.

        Returns
        -------
        set[frozenset]
            The associated set of C of vertices that follow rank(dual(matroid(G)),incidentEdges(C,G)) <= |C|
        """
        subsets = powerset(V)
        check_1 = set()
        for subset in subsets:
            if len(subset) == 0:
                continue
            B = self.edges_incident(subset, data)
            if dual.matroid_rank(B) <= len(subset):
                check_1.add(frozenset(subset))
        return check_1

    def _graph_curve_matroid(self, data: torch_geometric.data.Data) -> CCMatroid:
        """Algorithm proposed by Geiger et. al
        We compute the graph curve matroid by computing its circuits

        Parameters
        ----------
        data : torch_geometric.data.Data
            pytorch geometric data that contains data of a graph.

        Returns
        -------
        CCMatroid
            The associated graphic curve matroid of the graph in `data`.
        """
        graphic_matroid = self._generate_matroid_from_data(data)
        num_nodes = data.x.shape[0]
        dual = graphic_matroid.dual()

        groundset = list(range(num_nodes))
        rank_check_set = self.rank_check(dual, groundset, data)
        circuits = set()
        for C in rank_check_set:
            ok = True
            for A in powerset(C):
                A = frozenset(A)
                if A != C and A in rank_check_set:
                    ok = False
                    break
            if ok and len(C) != 0:
                circuits.add(C)

        return CCMatroid.from_circuits(ground=groundset, circuits=circuits)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        M_g = self._graph_curve_matroid(data)
        data = data.clone()
        data["ground"] = M_g.ground
        if self.max_rank:
            data["bases"] = M_g.skeleton(self.max_rank)
        else:
            data["bases"] = M_g.bases
        return data


class CurveLifting(Graph2CombinatorialLifting):
    r"""Implemented class for lifting graphs to combinatorial complexes via the graph curve matroid

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(
            GraphCurveMatroidLifting(**kwargs),
            Matroid2CombinatorialLifting(**kwargs),
            **kwargs,
        )
