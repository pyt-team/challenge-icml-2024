import networkx as nx
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import CombinatorialComplex

from modules.data.utils.utils import get_ccc_connectivity
from modules.transforms.liftings.lifting import AbstractLifting, GraphLifting
from modules.utils.matroid import *


class Graph2MatroidLifting(GraphLifting):
    r"""Abstract class for lifting graphs to matroids.
    Matroids are a special type of combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2matroid"

    def _generate_matroid_from_data(
        self, data: torch_geometric.data.Data
    ) -> GraphicMatroid:
        r"""Generates a graphic matroid from the input data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        GraphicMatroid
            The generated graphic matroid M(G).
        """
        edges = data.edge_index.t().tolist()
        return GraphicMatroid(edgelist=edges)

    def get_edges_incident(
        self, vertex: int | Iterable[int], data: torch_geometric.data.Data
    ):
        r"""Computes the edges incident by looking at data.edge_index

        Parameters
        ----------
        vertex : Iterable[Int]
            The input vertices to calculate the edges
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch.tensor
            The part of the edge_index that contains the vertex in question.
        """
        if not vertex:
            return torch.empty(0)
        row, col = data.edge_index
        vertices = (
            torch.tensor(vertex)
            if isinstance(vertex, Iterable)
            else torch.tensor(vertex).unsqueeze(0)
        )

        mask = (
            sum(row == vertex for vertex in vertices)
            .bool()
            .logical_or(sum(col == vertex for vertex in vertices))
        )
        ans = data.edge_index[:, mask]
        return ans


class Matroid2CombinatorialLifting(AbstractLifting):
    r"""Abstract class for lifting matroids to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "matroid2combinatorial"

    def matroid2cc(self, matroid: Matroid) -> CombinatorialComplex:
        rnk = matroid.rank
        cc_rank = lambda s: rnk(s) - 1
        cc = CombinatorialComplex()
        for ind in matroid.independent_sets:
            if len(ind) == 0:  # empty set isn't part of a CC
                continue

            cc.add_cell([i for i in ind], cc_rank(ind))
        return cc

    def _get_cell_attributes(
        self, cc: CombinatorialComplex, name: str, rank=None
    ) -> dict:
        attributes = cc.get_cell_attributes(name)
        if rank == None:
            return attributes

        return {ranked: attributes[ranked] for ranked in cc.skeleton(rank)}

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        ground = data["ground"]
        bases = data["bases"]
        M = Matroid(ground=ground, bases=bases)
        matroid_rank = M.rank(ground) - 1
        cc = self.matroid2cc(M)

        features = data["x"]
        rank_0_features = {
            node: features[list(node)].squeeze(0) for node in cc.skeleton(0)
        }
        cc.set_cell_attributes(rank_0_features, "features")

        connectivity = get_ccc_connectivity(cc, matroid_rank)
        # cc.get_cell_attributes("features", 0).values() doesn't seem to work? The alternative:
        connectivity["x_0"] = torch.stack(
            list(self._get_cell_attributes(cc, "features", 0).values())
        )
        return connectivity


class Graph2CombinatorialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to combinatorial complexes.

    Parameters
    ----------
    *liftings : the topological liftings needed to go from a graph to a combinatorial complex
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, *liftings, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2combinatorial"
        self.liftings: Iterable[AbstractLifting] = liftings

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a combinatorial complex. This is modified so that we can define multiple liftings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        if not self.liftings:
            raise NotImplementedError
        for lifting in self.liftings:
            data = lifting.lift_topology(data)
        return data
