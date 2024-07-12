"""Test the message passing module."""
import math

import networkx as nx
import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2pointcloud.spin_lifting import SpinLifting


class TestSpinLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the SimplicialCliqueLifting class
        self.spin_lifting = SpinLifting()

    def test_find_neighbors(self):
        """Test the find_neighbors method."""

        # Test the find_neighbors method
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        neighbors = self.spin_lifting.find_neighbors(graph, 0)
        assert neighbors == [1]
        neighbors = self.spin_lifting.find_neighbors(graph, 1)
        assert neighbors == [0, 2, 3]
        neighbors = self.spin_lifting.find_neighbors(graph, 2)
        assert neighbors == [1, 3]

    def test_calculate_coords_delta(self):
        """Test the calculate_coords_delta method."""

        # Test the calculate_coords_delta method
        allowable_error = 1e-10
        x_delta, y_delta = self.spin_lifting.calculate_coords_delta(30)
        assert x_delta - math.sqrt(3) / 2 < allowable_error
        assert y_delta - 0.5 < allowable_error
        x_delta, y_delta = self.spin_lifting.calculate_coords_delta(45)
        assert x_delta - math.sqrt(2) / 2 < allowable_error
        assert x_delta - math.sqrt(2) / 2 < allowable_error

    def test_assign_coordinates(self):
        """Test the assign_coordinates method."""

        # Test the assign_coordinates method
        allowable_error = 1e-10
        center_coords = (0, 0)
        neighbors = list(range(1, 14))
        coords_dict = self.spin_lifting.assign_coordinates(center_coords, neighbors)

        assert (
            coords_dict[1][0] - 1 < allowable_error
            and coords_dict[1][1] - 0.0 < allowable_error
        )
        assert (
            coords_dict[4][0] - 0.0 < allowable_error
            and coords_dict[4][1] - 1.0 < allowable_error
        )
        assert (
            coords_dict[13][0] - math.cos(math.radians(15)) < allowable_error
            and coords_dict[13][1] - math.sin(math.radians(15)) < allowable_error
        )

    def test_lift(self):
        """Test the lift method."""

        # Test the lift method
        allowable_error = 1e-10
        coords = {0: (0, 0)}
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        coords, max_distance = self.spin_lifting.lift(coords, graph, 0)
        print(coords)
        assert coords[1][0] - 1 < allowable_error and coords[1][1] - 0 < allowable_error
        assert coords[2][0] - 2 < allowable_error and coords[2][1] - 0 < allowable_error
        assert (
            coords[3][0] - (1 + math.sqrt(3) / 2) < allowable_error
            and coords[3][1] - (0 + 0.5) < allowable_error
        )
        assert max_distance - 2.0 < allowable_error

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data = self.spin_lifting.forward(self.data.clone())
        assert lifted_data.x.shape[0] == 8
        assert lifted_data.y.shape[0] == 8
        assert lifted_data.pos.shape == (8, 2)
        expected_pos = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [math.sqrt(3) / 2, 0.5],
                [1 + math.sqrt(3) / 2, 0.5],
                [0.5, math.sqrt(3) / 2],
                [math.sqrt(3), 1.0],
                [2 + math.sqrt(3) / 2, 0.5],
                [0.0, 1.0],
            ]
        )
        assert torch.allclose(lifted_data.pos, expected_pos)
