import pytest
import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2hypergraph.modularity_maximization_lifting import (
    ModularityMaximizationLifting,
)


class TestModularityMaximizationLifting:
    """Test the ModularityMaximizationLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()

        # Initialize the ModularityMaximizationLifting class
        self.lifting = ModularityMaximizationLifting(num_communities=2, k_neighbors=3)

    def test_lift_topology(self):
        # Set a random seed for reproducibility
        torch.manual_seed(42)

        # Test the lift_topology method
        lifted_data = self.lifting.lift_topology(self.data.clone())

        expected_n_hyperedges = self.data.num_nodes

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        assert (
            expected_incidence_1 == lifted_data["incidence_hyperedges"].to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data["num_hyperedges"]
        ), "Something is wrong with the number of hyperedges."


if __name__ == "__main__":
    pytest.main([__file__])
