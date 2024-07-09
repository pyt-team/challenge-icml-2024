import torch

from modules.data.utils.utils import load_manual_rings
from modules.transforms.liftings.graph2combinatorial.ring_close_atoms_lifting import (
    CombinatorialRingCloseAtomsLifting,
)


class TestCombinatorialRingCloseAtomsLifting:
    """Test the CombinatorialRingCloseAtomsLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_rings()

        # Initialise the CellCyclesLifting class
        self.lifting = CombinatorialRingCloseAtomsLifting()

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data = self.lifting.forward(self.data.clone())

        expected_num_hyperedges = 15

        assert (
            expected_num_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with num_hyperedges."

        expected_incidence_hyperedges = torch.tensor([
            [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0.],
            [0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ])

        assert (
            expected_incidence_hyperedges == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."

        expected_incidence_2 = torch.tensor([[0., 0.],
        [0., 0.],
        [1., 0.],
        [0., 0.],
        [1., 1.],
        [0., 1.],
        [1., 0.],
        [0., 0.],
        [0., 0.],
        [1., 0.],
        [0., 0.],
        [0., 1.],
        [0., 0.],
        [0., 0.]])

        assert (
            expected_incidence_2 == lifted_data.incidence_2.to_dense()
        ).all(), "Something is wrong with incidence_2."
