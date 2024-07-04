import networkx as nx
import torch_geometric
from toponetx.classes import CombinatorialComplex

from modules.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
    Graph2MatroidLifting,
    Matroid2CombinatorialLifting,
)
from modules.utils.matroid import *


class GraphCurveMatroidLifting(Graph2MatroidLifting):
    r"""Lifts graphs to graph curve matroids by identifying the cycles as 2-cells.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _graph_curve_matroid(self, data: torch_geometric.data.Data) -> Matroid:
        graphic_matroid = self._generate_matroid_from_data(data)
        num_nodes = data.x.shape[0]
        r_d = graphic_matroid.dual().rank
        d = lambda v: [
            tuple(edge)
            for edge in self.get_edges_incident(vertex=v, data=data).t().tolist()
        ]
        L = set()
        for C in powerset(range(num_nodes)):
            if len(C) == 0 or r_d(d(C)) > len(C):
                continue
            #  r_d(d(C)) <= len(C)
            subset_condition = True
            for A in powerset(C):
                a_size = len(A)
                if a_size == 0 or A == C:
                    continue
                if r_d(d(A)) <= a_size:
                    subset_condition = False
                    break
            if subset_condition:
                L.add(fs(C))
        return Matroid(list(range(num_nodes)), circuits_to_bases(L))

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        M_g = self._graph_curve_matroid(data)
        data = data.clone()
        data["ground"] = M_g._ground
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
