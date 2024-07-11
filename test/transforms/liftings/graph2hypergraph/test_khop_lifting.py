"""Test the message passing module."""

import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2hypergraph.knn_lifting import (
    HypergraphKNNLifting,
)


class TestHypergraphKHopLifting:
    """Test the HypergraphKHopLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the HypergraphKHopLifting class

        self.lifting_k = HypergraphKNNLifting(k_value=3)

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data_k = self.lifting_k.forward(self.data.clone())

        expected_n_hyperedges = 8

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        assert (
            expected_incidence_1 == lifted_data_k.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=1)."
        assert (
            expected_n_hyperedges == lifted_data_k.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=1)."
