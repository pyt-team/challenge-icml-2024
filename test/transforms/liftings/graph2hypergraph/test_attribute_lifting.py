import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.attribute_lifting import (
    NodeAttributeLifting,
)


class TestNodeAttributeLifting:
    """Test the NodeAttributeLifting class."""

    def setup_method(self):
        # Set up a simple manual graph for testing
        self.data = torch_geometric.data.Data(
            x=torch.tensor(
                [
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                ],
                dtype=torch.float,
            ),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
        )
        self.lifting = NodeAttributeLifting(attribute_idx=1)

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_topology = self.lifting.lift_topology(self.data.clone())

        expected_n_hyperedges = 2
        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ).to_sparse_coo()

        # Print for debugging
        print("Expected Incidence Matrix (Dense):")
        print(expected_incidence_1.to_dense())

        print("Actual Incidence Matrix (Dense):")
        print(lifted_topology["incidence_hyperedges"].to_dense())

        assert torch.equal(
            expected_incidence_1.to_dense(),
            lifted_topology["incidence_hyperedges"].to_dense(),
        ), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_topology["num_hyperedges"]
        ), "Something is wrong with the number of hyperedges."


# Running the test manually for debugging purposes
if __name__ == "__main__":
    test = TestNodeAttributeLifting()
    test.setup_method()
    test.test_lift_topology()
