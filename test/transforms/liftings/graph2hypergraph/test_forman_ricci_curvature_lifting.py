"""Test Forman-Ricci Curvature Lifting."""

import pytest
import torch
import torch.nn.functional as F
import torch_geometric

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2hypergraph.forman_ricci_curvature_lifting import (
    HypergraphFormanRicciCurvatureLifting,
)


class TestHypergraphFormanRicciCurvatureLifting:
    """Test the HypergraphFormanRicciCurvatureLifting class."""

    def setup_method(self):
        self.data = load_manual_graph()

        self.lifting = HypergraphFormanRicciCurvatureLifting(
            network_type="weighted",
            th_quantile=0.6,
        )

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data = self.lifting.forward(self.data.clone())

        expected_n_hyperedges = 2

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=1)."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=1)."
