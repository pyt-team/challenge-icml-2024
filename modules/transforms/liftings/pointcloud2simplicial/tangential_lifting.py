import gudhi as gd
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class TangentialLifting(PointCloud2SimplicialLifting):
    # intrinsic dimension of the manifold set to 1 by default
    def __init__(self, intrisic_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.intrisic_dim = intrisic_dim

    def _get_lifted_topology(self, simplicial_complex: SimplicialComplex) -> dict:
        lifted_topology = get_complex_connectivity(simplicial_complex, self.complex_dim)

        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )

        return lifted_topology

    def lift_topology(self, data: torch_geometric.data.Data, **kwargs) -> dict:

        # initialize tangential complex object
        tangential_complex = gd.TangentialComplex(self.intrisic_dim, data.pos)

        # build the complex
        tangential_complex.compute_tangential_complex()

        simplicial_complex = SimplicialComplex().from_gudhi(tangential_complex.create_simplex_tree())

        self.complex_dim = simplicial_complex.dim

        node_features = {i: data.x[i, :] for i in range(data.x.shape[0])}
        simplicial_complex.set_simplex_attributes(node_features, name="features")

        return self._get_lifted_topology(simplicial_complex)
