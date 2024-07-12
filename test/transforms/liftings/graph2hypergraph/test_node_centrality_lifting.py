"""Test Page Rank Lifting."""

import pytest
import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2hypergraph.node_centrality_lifting import (
    HypergraphNodeCentralityLifting,
)


class TestHypergraphNodeCentralityLifting:
    """Test the HypergraphNodeCentralityLifting class."""

    def setup_method(self):
        self.data = load_manual_graph()

        self.lifting = HypergraphNodeCentralityLifting(
            network_type="weighted",
            th_percentile=0.2,
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
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges."

        self.lifting.network_type = "unweighted"
        lifted_data = self.lifting.forward(self.data.clone())

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges."

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [1.0, 0.0],
            ]
        )

        self.lifting.network_type = "unweighted"
        self.lifting.do_weight_hyperedge_influence = True
        lifted_data = self.lifting.forward(self.data.clone())

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges."

        assert (
            lifted_data.x_hyperedges.to_dense() == torch.tensor([[5106.0], [1060.0]])
        ).all(), "Something is wrong with x_hyperedges."

        self.lifting.do_hyperedge_node_assignment_feature_lifting_passthrough = True
        lifted_data = self.lifting.forward(self.data.clone())

        assert (
            lifted_data.x_hyperedges.to_dense() == torch.tensor([[1.0], [10.0]])
        ).all(), "Something is wrong with x_hyperedges."

    def test_validations(self):
        with pytest.raises(NotImplementedError):
            self.lifting.network_type = "mixed"
            self.lifting.forward(self.data.clone())
