import torch_geometric

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
        r_d = graphic_matroid.dual().matroid_rank

        def d(v):
            return [
                tuple(edge)
                for edge in self.get_edges_incident(vertex=v, data=data).t().tolist()
            ]

        L = set()

        for C in powerset(range(num_nodes)):
            sizeC = len(C)
            if sizeC == 0 or r_d(d(C)) > sizeC:
                continue
            subset_condition = True
            for A in powerset(C):
                a_size = len(A)
                if a_size == 0 or A == C:
                    continue
                if r_d(d(A)) <= a_size:
                    subset_condition = False
                    break
            if subset_condition:
                L.add(frozenset(C))
        groundset = range(num_nodes)
        return CCMatroid.from_circuits(ground=groundset, circuits=L)

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
