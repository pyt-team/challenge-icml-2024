"Test the ProjectionSum feature lifting."

import torch

from modules.io.load.loaders import manual_simple_graph
from modules.transforms.feature_liftings.feature_liftings import ProjectionSum
from modules.transforms.liftings.graph2hypergraph.khop_lifting import (
    HypergraphKHopLifting,
)
from modules.transforms.liftings.graph2simplicial.clique_lifting import (
    SimplicialCliqueLifting,
)


class TestProjectionSum:
    """Test the ProjectionSum class."""

    def setup_method(self):
        # Load the graph
        self.data = manual_simple_graph()
        # Initialize the ProjectionLifting class
        self.feature_lifting = ProjectionSum()

        # Initialize a simplicial/cell lifting class
        self.lifting = SimplicialCliqueLifting(complex_dim=3)
        # Initialize a hypergraph lifting class
        self.lifting_h = HypergraphKHopLifting(k_value=1)

    def test_lift_features(self):
        # Test the lift_features method for simplicial/cell lifting
        lifted_data = self.lifting.forward(self.data.clone())
        del lifted_data.x_1
        del lifted_data.x_2
        del lifted_data.x_3
        lifted_data = self.feature_lifting.forward(lifted_data)

        expected_x1 = torch.tensor(
            [
                [6.0],
                [11.0],
                [101.0],
                [5001.0],
                [15.0],
                [105.0],
                [60.0],
                [110.0],
                [510.0],
                [5010.0],
                [1050.0],
                [1500.0],
                [5500.0],
            ]
        )

        expected_x2 = torch.tensor(
            [[32.0], [212.0], [222.0], [10022.0], [230.0], [11020.0]]
        )

        expected_x3 = torch.tensor([[696.0]])

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."

        # Test the lift_features method for hypergraph lifting
        lifted_data = self.lifting_h.forward(self.data.clone())
        del lifted_data.x_hyperedges
        lifted_data = self.feature_lifting.forward(lifted_data)

        expected_x_0 = torch.tensor(
            [[1.0], [5.0], [10.0], [50.0], [100.0], [500.0], [1000.0], [5000.0]]
        )

        expected_x_hyperedges = torch.tensor(
            [
                [5116.0],
                [116.0],
                [5666.0],
                [1060.0],
                [116.0],
                [6510.0],
                [1550.0],
                [5511.0],
            ]
        )

        assert (
            expected_x_0 == lifted_data.x_0
        ).all(), "Something is wrong with the lifted features x_0."

        assert (
            expected_x_hyperedges == lifted_data.x_hyperedges
        ).all(), "Something is wrong with the lifted features x_hyperedges."
