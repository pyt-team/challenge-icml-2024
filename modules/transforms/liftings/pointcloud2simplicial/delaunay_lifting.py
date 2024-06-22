import torch_geometric
from scipy.spatial import Delaunay
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class DelaunayLifting(PointCloud2SimplicialLifting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        tri = Delaunay(data)  # data.pos
        simplices = tri.simplices

        simplicial_complex = SimplicialComplex(simplices)
        return simplicial_complex
