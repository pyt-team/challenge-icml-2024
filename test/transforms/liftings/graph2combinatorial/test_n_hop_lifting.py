"""Test the message passing module."""

import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2combinatorial.n_hop_lifting import (
    NHopLifting,
)


class TestNHopLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the NHopLifting class
        self.lifting_signed = NHopLifting(hop_num=1)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data = self.lifting_signed.forward(self.data.clone())

        expected_incidence_hyperedge = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )

        print(lifted_data.incidence_hyperedges.to_dense())
        assert (
            abs(expected_incidence_hyperedge) == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedge (nodes to hyperedges)."
