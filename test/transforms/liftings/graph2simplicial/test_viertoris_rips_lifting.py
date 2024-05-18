"""Test the message passing module."""

import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2simplicial.vietoris_rips_lift import(
    SimplicialVietorisRipsLifting,
)


class TestSimplicialVietorisRipsLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the SimplicialCliqueLifting class
        self.lifting_signed = SimplicialVietorisRipsLifting(complex_dim=2, dis=2, signed=True)
        self.lifting_unsigned = SimplicialVietorisRipsLifting(complex_dim=3, dis=2, signed=False)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        # TODO CHange
        expected_incidence_1 = torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_1) == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), "Something is wrong with unsigned incidence_1 (nodes to edges)."
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        # TODO change
        expected_incidence_2 = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_2) == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), "Something is wrong with signed incidence_2 (edges to triangles)."

        expected_incidence_3 = torch.tensor(
            [[-1.0], [1.0], [-1.0], [0.0], [1.0], [0.0]]
        )

        assert (
            abs(expected_incidence_3) == lifted_data_unsigned.incidence_3.to_dense()
        ).all(), (
            "Something is wrong with unsigned incidence_3 (triangles to tetrahedrons)."
        )
        assert (
            expected_incidence_3 == lifted_data_signed.incidence_3.to_dense()
        ).all(), (
            "Something is wrong with signed incidence_3 (triangles to tetrahedrons)."
        )
