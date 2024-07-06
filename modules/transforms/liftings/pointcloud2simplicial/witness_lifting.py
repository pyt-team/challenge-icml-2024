import gudhi as gd
import torch
import torch_geometric

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class WitnessLifting(PointCloud2SimplicialLifting):
    def __init__(self, is_weak=True, is_euclidian=True, **kwargs):
        super().__init__(**kwargs)
        self.is_weak = is_weak
        self.is_euclidian = True  # is_euclidian

    def lift_topology(
        self,
        witnesses: torch_geometric.data.Data,
        landmarks: torch_geometric.data.Data = None,
        landmark_proportion: int = 0.15,
    ) -> dict:
        n = len(witnesses.pos)
        if not landmarks:
            perm = torch.randperm(n)
            idx = perm[: round(n * landmark_proportion)]
            landmarks_position = witnesses.pos[idx]
            print(landmarks_position)
        else:
            landmarks_position = landmarks.pos

        if self.is_euclidian:
            if self.is_weak:
                complex = gd.EuclideanWitnessComplex(witnesses.pos, landmarks_position)
                pass
            else:
                pass
