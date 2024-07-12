import torch
from torch_geometric.data import Data

from modules.transforms.liftings.pointcloud2graph.delaunay_lifting import (
    GraphDelaunayLifting,
)


class TestDelaunayLifting:
    """Test the GraphDelaunayLifting class."""

    def setup_method(self):
        """Set up the test."""
        # Define the data and the lifting.
        pos = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.5],
                [1.0, 1.0, 0.5],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )
        x = torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
            dtype=torch.float32,
        )
        self.data = Data(x=x, pos=pos)
        self.lifting = GraphDelaunayLifting()

    def test_lift_topology(self):
        """Test the lift_topology method."""

        lifted = self.lifting.forward(self.data.clone())
        assert lifted.num_nodes == 5, "The number of nodes is incorrect."
        assert lifted.edge_index.shape == (2, 14), "The number of edges is incorrect."
        assert lifted.face.shape == (4, 2), "The number of faces is incorrect."
