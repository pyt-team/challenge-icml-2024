import pudb
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting
from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


class GraphKNNLifting(PointCloud2GraphLifting, Graph2SimplicialLifting):
    r"""Lifts point cloud data to graph by creating its k-NN graph

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts a point cloud dataset to a graph by constructing its k-NN graph.

        Parameters
        ----------
        data :  torch_geometric.data.Data
            The input data to be lifted

        Returns
        -------
        dict
            The lifted topology
        """
        # Pull in node features
        transform = torch_geometric.transforms.KNNGraph()
        graph_data = transform(data)
        graph = self._generate_graph_from_data(graph_data)
        # Create dummy simplicial complex
        simplicial_complex = SimplicialComplex(graph)
        return self._get_lifted_topology(simplicial_complex, graph)
