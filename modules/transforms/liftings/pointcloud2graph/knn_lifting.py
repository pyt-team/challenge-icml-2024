import torch
import torch_geometric

from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


class GraphKNNLifting(PointCloud2GraphLifting):
    r"""Lifts point cloud data to graph by creating its k-NN graph

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class
    """

    def __init__(self, k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.transform = torch_geometric.transforms.KNNGraph(k=k)

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
        graph_data = self.transform(data)
        topology = {
            "shape": [graph_data.x.shape[0], graph_data.edge_index.shape[1]],
            "edge_index": graph_data.edge_index,
            "num_nodes": graph_data.x.shape[0],
        }
        return topology
