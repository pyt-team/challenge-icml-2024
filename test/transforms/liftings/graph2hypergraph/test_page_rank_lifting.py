"""Test Page Rank Lifting."""

import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2hypergraph.page_rank_lifting import (
    HypergraphPageRankLifting,
)


class TestHypergraphPageRankLifting:
    """Test the HypergraphPageRankLifting class."""

    def setup_method(self):
        self.data = load_manual_graph()

        self.lifting = HypergraphPageRankLifting(
            network_type="weighted",
            alpha=0.85,
            th_quantile=0.8,
            n_most_influential=1,
        )

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data = self.lifting.forward(self.data.clone())

        expected_n_hyperedges = 2

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=1)."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=1)."
