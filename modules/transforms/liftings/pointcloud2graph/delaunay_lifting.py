import torch
import torch_geometric
from torch_geometric.transforms import Delaunay
from torch_geometric.utils import to_undirected

from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


class GraphDelaunayLifting(PointCloud2GraphLifting):
    r"""Lifts point cloud to graph domain by considering k-nearest neighbors.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = Delaunay()

    def face_to_edge(self, data: torch_geometric.data.Data):
        r"""Converts mesh faces to edges indices for both 2D and 3D meshes.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be converted.

        Returns
        -------
        torch_geometric.data.Data
            The converted data.
        """
        if hasattr(data, "face"):
            assert data.face is not None
            face = data.face
            if face.shape[0] == 3:
                # 2D
                edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
                edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
            elif face.shape[0] == 4:
                # 3D
                edge_index = torch.cat(
                    [face[:2], face[1:3], face[2:], face[::3]], dim=1
                )
                edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
            else:
                raise ValueError("Faces must be of dimension 2 or 3.")
            data.edge_index = edge_index
        return data

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain by considering k-nearest neighbors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        num_nodes = data.x.shape[0]

        # Step 1: Perform Delaunay Triangulation to get faces
        data_delaunay = self.transform(data)
        faces = data_delaunay.face
        # Step 2: Create Edge List from faces
        data = self.face_to_edge(data_delaunay)

        # Step 3: Convert Edge List to edge_index format
        edge_index = data.edge_index

        return {"num_nodes": num_nodes, "edge_index": edge_index, "face": faces}
