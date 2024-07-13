"""Test the feature-based Rips complex lifting."""

import torch
import torch_geometric

from modules.data.utils.utils import load_manual_points
from modules.transforms.liftings.pointcloud2simplicial.feature_rips_complex_lifting import (
    FeatureRipsComplexLifting,
)


class TestFeatureRipsComplexLifting:
    """Test the FeatureRipsComplexLifting class."""

    def setup_method(self):
        # Load the point cloud
        self.data = load_manual_points()

        # Initialise the FeatureRipsLifting class
        self.position_lifting = FeatureRipsComplexLifting(
            complex_dim=3, feature_percent=0.0, max_edge_length=10.0
        )
        self.feature_lifting = FeatureRipsComplexLifting(
            complex_dim=3, feature_percent=1.0, max_edge_length=1.0
        )
        self.mixed_lifting = FeatureRipsComplexLifting(
            complex_dim=3, feature_percent=0.2, max_edge_length=10.0
        )
        self.no_epsilon_lifting = FeatureRipsComplexLifting(
            complex_dim=3,
            feature_percent=0.5,
            max_edge_length=10.0,
            epsilon=0.0,
        )

    def test_generate_distance_matrix(self):
        # Test the generate_distance_matrix method
        data = torch_geometric.data.Data(
            pos=torch.tensor([[-1], [1], [2]]).float(),
            x=torch.tensor([[2], [4], [6]]).float(),
        )
        expected_pairwise_distances = torch.tensor(
            [[0, 2, 3.5], [2, 0, 1.5], [3.5, 1.5, 0]]
        )

        assert (
            self.no_epsilon_lifting.generate_distance_matrix(data)
            == expected_pairwise_distances
        ).all(), "generate_distance_matrix not working as expected"

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data = self.position_lifting.forward(self.data.clone())

        expected_incidences = (
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0]]),
        )

        assert (
            expected_incidences[0] == lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with incidence_1 (nodes to edges) for feature_percent=0.0."

        assert (
            abs(expected_incidences[1]) == lifted_data.incidence_2.to_dense()
        ).all(), "Something is wrong with incidence_2 (edges to triangles) for feature_percent=0.0."

        assert (
            abs(expected_incidences[2]) == lifted_data.incidence_3.to_dense()
        ).all(), "Something is wrong with incidence_3 (triangles to tetrahedrons) for feature_percent=0.0."

        expected_incidences = (
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
        )
        lifted_data = self.feature_lifting.forward(self.data.clone())

        assert (
            expected_incidences[0] == lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with incidence_1 (nodes to edges) for feature_percent=1.0."

        assert (
            abs(expected_incidences[1]) == lifted_data.incidence_2.to_dense()
        ).all(), "Something is wrong with incidence_2 (edges to triangles) for feature_percent=1.0."

        assert (
            abs(expected_incidences[2]) == lifted_data.incidence_3.to_dense()
        ).all(), "Something is wrong with incidence_3 (triangles to tetrahedrons) for feature_percent=1.0."

        expected_incidences = (
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            ),
        )

        lifted_data = self.mixed_lifting.forward(self.data.clone())

        assert (
            expected_incidences[0] == lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with incidence_1 (nodes to edges) for feature_percent=0.2 ."

        assert (
            abs(expected_incidences[1]) == lifted_data.incidence_2.to_dense()
        ).all(), "Something is wrong with incidence_2 (edges to triangles) for feature_percent=0.2 ."

        assert (
            abs(expected_incidences[2]) == lifted_data.incidence_3.to_dense()
        ).all(), "Something is wrong with incidence_3 (triangles to tetrahedrons) for feature_percent=0.2 ."
