"""Test the message passing module."""

import torch
from modules.data.utils.utils import load_sphere_point_cloud
from modules.transforms.liftings.pointcloud2simplicial.ball_pivoting_lifting import BallPivotingLifting


class TestBallPivotingLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Create the point cloud
        self.sphere_pc = load_sphere_point_cloud(num_classes=2, num_points=3, num_features=1, seed=0)
        radii = [0.1, 2.0]

        # Initialize the BallPivotingLifting class
        self.lifting = BallPivotingLifting(radii=radii)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data = self.lifting.forward(self.sphere_pc)

        expected_incidence_0_indices = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 2]
        ])

        assert(
            torch.all(abs(expected_incidence_0_indices) == lifted_data["incidence_0"].indices())
        )

        expected_incidence_1_indices = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 2, 1, 2]
        ])

        assert(
            torch.all(abs(expected_incidence_1_indices) == lifted_data["incidence_1"].indices())
        ), "Something is wrong with the incidence_1 matrix (nodes to edges)."

        expected_incidence_2_indices = torch.tensor(
        [
           [0, 1, 2],
           [0, 0, 0]
        ])

        assert(
            torch.all(abs(expected_incidence_2_indices) == lifted_data["incidence_2"].indices())
        ), "Something is wrong with the incidence_2 matrix (edges to triangles)."
