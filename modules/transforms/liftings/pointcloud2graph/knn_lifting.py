import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


class GraphKNNLifting(PointCloud2GraphLifting):
    r"""Lifts point cloud data to graph by creating its k-NN graph

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class
    """

    def __init__(self, **kwargss):
        super.__init__(**kwargs)

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
        transform = torch_geometric.transforms.KNNGraph(**self.kwargs)
        graph_data = transform(data)
        # Create dummy simplicial complex
        simplicial_complex = SimplicialComplex(graph)

        # Hack around, create a 1-simplicial complex
        lifted_topology = get_complex_connectivity(cell_complex, 1)
        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_cell_attributes("features", 0).values())
        )
        return lifted_topology
