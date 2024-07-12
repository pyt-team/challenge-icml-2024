import torch_geometric
import open3d
import numpy as np
import warnings
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class BallPivotingLifting(PointCloud2SimplicialLifting):
    """Uses the Ball Pivoting Algorithm to lift an input point cloud to a simplical complex.
        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.
        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data."""

    def __init__(self, radii: list[float], **kwargs):
        super().__init__(**kwargs)
        self.radii = radii

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        # Convert input data into an open3d point cloud
        open3d_point_cloud = open3d.geometry.PointCloud(open3d.cpu.pybind.utility.Vector3dVector(data.pos.numpy()))
        # Check that the input point cloud includes normals. The Ball Pivoting Algorithm requires normals.
        if "normals" not in data:
            warnings.warn("Normals not found in data set. The Ball Pivoting algorithm requires oriented 3D points, thus, normals will be estimated using the 'estimate_normals' method. Note, the normals are often not estimated with great success, so the performance of the algorithm might suffer heavily from this.")
    
            open3d_point_cloud.estimate_normals()
        else:
            open3d_point_cloud.normals = open3d.cpu.pybind.utility.Vector3dVector(data.normals.numpy())

        rec_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        open3d_point_cloud, open3d.utility.DoubleVector(self.radii))

        # Convert output to proper format
        simplices = np.asarray(rec_mesh.triangles)

        simplicial_complex = SimplicialComplex(simplices)

        lifted_topology = self._get_lifted_topology(simplicial_complex)
        lifted_topology["x_0"] = data.x

        return lifted_topology
    
    def _get_lifted_topology(self, simplicial_complex: SimplicialComplex) -> dict:
        r"""Returns the lifted topology.
        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.
        Returns
        ---------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(simplicial_complex, self.complex_dim)
        return lifted_topology
    
    def plot_lifted_topology(self, data: torch_geometric.data.Data):
        # Convert input data into an open3d point cloud
        open3d_point_cloud = open3d.geometry.PointCloud(open3d.cpu.pybind.utility.Vector3dVector(data.pos.numpy()))
        # Check that the input point cloud includes normals. The Ball Pivoting Algorithm requires normals.
        if "normals" not in data:
            warnings.warn("Normals not found in data set. The Ball Pivoting algorithm requires oriented 3D points, thus, normals will be estimated using the 'estimate_normals' method. Note, the normals are often not estimated with great success, so the performance of the algorithm might suffer heavily from this.")
    
            open3d_point_cloud.estimate_normals()
        else:
            open3d_point_cloud.normals = open3d.cpu.pybind.utility.Vector3dVector(data.normals.numpy())

        rec_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        open3d_point_cloud, open3d.utility.DoubleVector(self.radii))
        open3d.visualization.draw_geometries([open3d_point_cloud, rec_mesh])
