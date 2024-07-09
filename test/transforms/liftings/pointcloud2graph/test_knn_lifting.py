import torch

from modules.data.utils.utils import load_manual_prot_pointcloud
from modules.transforms.liftings.pointcloud2graph.knn_lifting import (
    PointCloudKNNLifting,
)


class TestPointCloudKNNLifting:
    """Test the PointCloudKNNLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_prot_pointcloud()

        # Initialise the CellCyclesLifting class
        self.lifting = PointCloudKNNLifting()

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data = self.lifting.forward(self.data.clone())

        expected_num_nodes = 16

        assert expected_num_nodes == lifted_data.num_nodes, "Something is wrong with the number of nodes."

        expected_edge_index = torch.tensor([
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  4,  4,  5,
          8,  5,  8, 10,  0, 11, 11, 13, 15,  6,  7,  6,  7,  4,  3,  5,  8,  9,
         14,  5,  8,  9,  0,  2, 11,  1, 13,  6, 15,  7,  6,  7,  3, 14,  8,  9,
          5,  9,  0, 11,  1, 15,  7,  7,  3, 12,  3,  5, 14,  9,  0,  9,  1, 13,
         10, 13, 15, 12,  5, 12, 14,  5,  9, 10,  1, 13,  6,  7, 15,  7, 12,  3,
          5, 14,  9,  5, 10,  8,  1, 13,  2, 15,  7,  7,  6, 12,  3, 14,  4,  9,
          5,  8, 10,  8, 10,  1, 13,  2,  7,  6,  3,  3,  5,  4, 14, 10,  1,  0,
         10, 11,  2,  6, 12,  4, 12, 14,  4,  8,  8, 10,  1,  0,  2, 11, 11,  6,
          6, 12,  3,  8, 10,  8,  1,  0,  2, 11,  9, 15,  6,  7, 12, 14,  4,  8,
         10,  9,  0,  5,  8, 10,  9,  0,  2, 15,  6,  7,  4,  5, 10,  9,  0,  2,
         10,  9, 11,  2, 13,  6, 13, 15,  6,  7],
        [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  9,  1,
          0, 10,  9,  6,  5,  5, 14,  8,  5,  2,  1, 11, 10,  2,  6,  3,  2,  1,
         15, 12, 11, 10,  7,  4,  7,  8, 10,  4,  7,  3, 13, 12,  8,  8,  4,  3,
         14, 12,  9,  9, 10,  9,  5, 14,  1, 13, 10,  7, 10,  5,  2, 14,  3,  5,
         15, 14, 11,  6,  0, 15, 12,  9,  7,  8,  5,  7,  1,  0, 13,  9,  8,  5,
          2,  5,  0, 11,  1, 13,  7,  9,  6,  6,  2, 11, 15, 10,  7,  7,  6,  2,
         13,  6,  3, 15, 12,  9, 11,  8,  4,  8,  0,  9,  6,  8,  9,  5,  2,  4,
         14, 13, 10, 10,  5,  1, 14, 11, 10,  1, 10,  7,  4,  6,  3,  6, 15,  3,
         12,  7,  4,  3,  0, 12,  6,  8,  5,  8, 11,  8, 14, 13,  9,  6,  5,  5,
          2,  4,  1, 15, 14, 11, 13, 10,  7, 10,  7, 15,  7,  8,  4,  6,  3,  0,
         13, 15, 12,  9,  6,  0, 15, 12,  9,  8]])


        assert (
            expected_edge_index == lifted_data.edge_index.to_dense()
        ).all(), "Something is wrong with edge_index."

