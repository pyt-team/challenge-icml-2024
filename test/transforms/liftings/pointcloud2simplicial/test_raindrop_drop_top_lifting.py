import torch
import torch_geometric

from modules.transforms.liftings.pointcloud2simplicial.raindrop_drop_top_lifting import (
    RaindropDropTopLifting,
)


class TestRaindropDropTopLifting:
    def setup_method(self):
        self.lifting = RaindropDropTopLifting(
            n_sens=10,
            d_model=64,
            n_layers=2,
            n_heads=4,
            drop_ratio=0.4,
            n_classes=2,
            d_static=5,
            max_len=215,
            threshold=0.2,
            epoch=1,
        )
        self.data = torch_geometric.data.Data(
            x=torch.randn(100, 64),
            times=torch.randn(100, 1),
            static=torch.randn(100, 5),
            y=torch.randint(0, 2, (100,)),
        )

    def test_lift_topology(self):
        simplex = self.lifting.lift_topology(self.data)
        assert isinstance(simplex, dict), "The output should be a dictionary."
        assert (
            "incidence_matrices" in simplex
        ), "The output dictionary should contain incidence matrices."
