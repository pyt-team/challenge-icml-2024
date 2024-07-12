import gudhi as gd
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class WitnessLifting(PointCloud2SimplicialLifting):
    def __init__(
        self,
        is_weak=True,
        is_euclidian=True,
        landmark_proportion: int = 0.8,
        max_alpha_square=0.15,
        complex_dim=2,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_weak = is_weak
        self.is_euclidian = is_euclidian
        self.landmark_proportion = landmark_proportion
        self.max_alpha_square = max_alpha_square
        self.complex_dim = complex_dim
        self.seed = seed
        torch.manual_seed(seed)

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
        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )

        return lifted_topology

    def lift_topology(
        self,
        witnesses: torch_geometric.data.Data,
    ) -> dict:
        n = len(witnesses.pos)

        perm = torch.randperm(n)
        idx = perm[: round(n * self.landmark_proportion)]
        landmarks_position = witnesses.pos[idx]

        if self.is_euclidian:
            if self.is_weak:
                complex = gd.EuclideanWitnessComplex(witnesses.pos, landmarks_position)
                simplex_tree = complex.create_simplex_tree(
                    self.max_alpha_square, self.complex_dim
                )

                simplicial_complex = SimplicialComplex.from_gudhi(simplex_tree)
            else:
                pass

        node_features = {i: witnesses.x[i, :] for i in range(witnesses.x.shape[0])}
        simplicial_complex.set_simplex_attributes(node_features, name="features")

        return self._get_lifted_topology(simplicial_complex)
