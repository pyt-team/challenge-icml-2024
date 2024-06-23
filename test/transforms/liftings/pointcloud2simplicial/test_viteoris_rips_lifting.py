import unittest

import torch
from torch_geometric.data import Data

from modules.transforms.liftings.pointcloud2simplicial.vietoris_rips_lifting import (
    VietorisRipsLifting,
)
from modules.utils.utils import describe_data


class TestVietorisRipsLifting(unittest.TestCase):
    def setUp(self):
        # Set up some basic point cloud data for testing
        self.points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float
        )
        self.features = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
        self.data = Data(pos=self.points, x=self.features)
        self.epsilon = 1.5  # Set epsilon distance

    def test_initialization(self):
        # Test initialization
        lifting = VietorisRipsLifting(epsilon=self.epsilon)
        self.assertEqual(lifting.epsilon, self.epsilon)

    def test_lift_topology(self):
        # Test lift_topology method
        lifting = VietorisRipsLifting(epsilon=self.epsilon)
        lifted_topology = lifting.lift_topology(self.data)

        # Check if the lifted topology contains expected keys
        self.assertIn("x_0", lifted_topology)

        # Check if the number of vertices matches the input points
        self.assertEqual(lifted_topology["shape"][0], len(self.points))

        # Check if the features are correctly assigned
        for i, feature in enumerate(self.features):
            self.assertTrue(torch.equal(lifted_topology["x_0"][i], feature))

    def test_lifted_topology_structure(self):
        # Check the structure of the lifted topology
        lifting_tiny_epsilon = VietorisRipsLifting(epsilon=0.5)
        lifted_topology_tiny = lifting_tiny_epsilon.lift_topology(self.data)

        # Ensure the output is a dictionary
        self.assertIsInstance(lifted_topology_tiny, dict)

        self.assertEqual(lifted_topology_tiny["shape"], [4])

        lifting_small_epsilon = VietorisRipsLifting(epsilon=1)
        lifted_topology_small = lifting_small_epsilon.lift_topology(self.data)
        self.assertEqual(lifted_topology_small["shape"], [4, 4])

        lifting_large_epsilon = VietorisRipsLifting(epsilon=1.5)
        lifted_topology_large = lifting_large_epsilon.lift_topology(self.data)
        self.assertEqual(lifted_topology_large["shape"], [4, 6, 4, 1])

    def test_epsilon_effect(self):
        # Test the effect of different epsilon values
        lifting_small_epsilon = VietorisRipsLifting(epsilon=1)
        lifted_topology_small = lifting_small_epsilon.lift_topology(self.data)
        simplices_count_small = sum(lifted_topology_small["shape"])

        lifting_large_epsilon = VietorisRipsLifting(epsilon=1.5)
        lifted_topology_large = lifting_large_epsilon.lift_topology(self.data)
        simplices_count_large = sum(lifted_topology_large["shape"])

        # With a smaller epsilon, the total number of simplic
        self.assertGreater(simplices_count_large, simplices_count_small)


if __name__ == "__main__":
    unittest.main()
