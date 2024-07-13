"""Test the message passing module."""

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2hypergraph.spectral_lifting import (
    SpectralLifting,
)


class TestSpectralLifting:
    """Test the SpectralLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the SpectralLifting class
        self.num_hyperedges = 3
        self.lifting = SpectralLifting(n_c=self.num_hyperedges)

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data_k = self.lifting.forward(self.data.clone())

        assert list(lifted_data_k.incidence_hyperedges.shape) == [
            self.data.num_nodes,
            self.num_hyperedges,
        ], "There were issues computing the incidence_hyperedges matrix"
