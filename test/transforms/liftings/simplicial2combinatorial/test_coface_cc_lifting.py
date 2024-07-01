"""Test the message passing module."""


import torch

from modules.data.utils.utils import load_manual_simplicial_complex
from modules.transforms.liftings.simplicial2combinatorial.coface_cc_lifting import (
    CofaceCCLifting,
)


class TestCofaceCCLifting:
    """Test the CofaceCCLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_simplicial_complex()

        # Initialise the CofaceCCLifting class
        self.coface_lift = CofaceCCLifting()

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data = self.coface_lift.forward(self.data.clone())

        expected_n_3_cells = 3

        print(lifted_data.incidence_3.to_dense())
        expected_incidence_3 = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        assert (
            expected_incidence_3 == lifted_data.incidence_3.to_dense()
        ).all(), "Something is wrong with incidence_3 ."
        assert (
            expected_n_3_cells == lifted_data.x_3.size(0)
        ), "Something is wrong with the number of 3-cells."
