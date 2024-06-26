import gudhi as gd
import torch_geometric

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class WitnessLifting(PointCloud2SimplicialLifting):
    def __init__(self, is_weak=True, is_euclidian=True, **kwargs):
        super().__init__(**kwargs)
        self.is_weak = is_weak
        self.is_euclidian = is_euclidian

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        pass
