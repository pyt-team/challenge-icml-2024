import torch
import torch_geometric

from modules.transforms.liftings.pointcloud2simplicial.raindrop_drop_top_lifting import (
    RaindropDropTopLifting,
)


class TestRaindropDropTopLifting:
    def setup_method(self):
        self.lifting = RaindropDropTopLifting(
            n_sens=4,
            d_model=8,
            n_layers=2,
            n_heads=1,
            drop_ratio=0.4,
            n_classes=2,
            d_static=9,
            max_len=215,
            threshold=0.2,
            epoch=1,
        )
        self.data = torch_geometric.data.Data(
            x=torch.randn(215, 16, 4),
            times=torch.randn(215, 16),
            static=torch.randn(16, 9),
            y=torch.randint(0, 2, (16,)),
        )

    def test_lift_topology(self):
        simplex = self.lifting.lift_topology(self.data)
        assert isinstance(simplex, dict), "The output should be a dictionary."
        assert (
            "incidence_matrices" in simplex
        ), "The output dictionary should contain incidence matrices."
