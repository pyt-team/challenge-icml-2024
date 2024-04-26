"""Test the message passing module."""
import pytest

import torch
from modules.io.load.loaders import manual_simple_graph
from modules.transforms.feature_liftings.feature_liftings import (
    SumLifting,
)
from modules.transforms.liftings.graph2hypergraph.khop_lifting import HypergraphKHopLifting

class TestSumLifting:
    """Test the SumLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = manual_simple_graph()

        # Initialize a lifting class
        self.lifting = HypergraphKHopLifting(k_value=1)
        # Initialize the ProjectionLifting class
        self.feature_lifting = SumLifting()

    def test_lift_features(self):
        # Test the lift_features method
        lifted_data = self.lifting.forward(self.data.clone())
        del lifted_data.x_1
        del lifted_data.x_2
        del lifted_data.x_3
        lifted_data = self.feature_lifting.forward(lifted_data)

        expected_x_0 = torch.tensor(
            [
                [   1.],
                [   5.],
                [  10.],
                [  50.],
                [ 100.],
                [ 500.],
                [1000.],
                [5000.]
            ]
        )

        expected_x_hyperedges = torch.tensor(
            [
                [5116.],
                [ 116.],
                [5666.],
                [1060.],
                [ 116.],
                [6510.],
                [1550.],
                [5511.]
            ]
        )


        assert (
            expected_x_0 == lifted_data.x_0
        ).all(), "Something is wrong with the lifted features x_0."
        
        assert (
            expected_x_hyperedges == lifted_data.x_hyperedges
        ).all(), "Something is wrong with the lifted features x_hyperedges."
       