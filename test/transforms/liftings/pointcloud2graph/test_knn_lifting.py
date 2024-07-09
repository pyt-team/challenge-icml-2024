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

        expected_edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                            [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [4, 0], [4, 9],
                            [5, 1], [8, 0], [5, 10], [8, 9], [10, 6], [0, 5], [11, 5], [11, 14], [13, 8],
                            [15, 5], [6, 2], [7, 1], [6, 11], [7, 10], [4, 2], [3, 6], [5, 3], [8, 2],
                            [9, 1], [14, 15], [5, 12], [8, 11], [9, 10], [0, 7], [2, 4], [11, 7], [1, 8],
                            [13, 10], [6, 4], [15, 7], [7, 3], [6, 13], [7, 12], [3, 8], [14, 8], [8, 4],
                            [9, 3], [5, 14], [9, 12], [0, 9], [11, 9], [1, 10], [15, 9], [7, 5], [7, 14],
                            [3, 1], [12, 13], [3, 10], [5, 7], [14, 10], [9, 5], [0, 2], [9, 14], [1, 3],
                            [13, 5], [10, 15], [13, 14], [15, 11], [12, 6], [5, 0], [12, 15], [14, 12],
                            [5, 9], [9, 7], [10, 8], [1, 5], [13, 7], [6, 1], [7, 0], [15, 13], [7, 9],
                            [12, 8], [3, 5], [5, 2], [14, 5], [9, 0], [5, 11], [10, 1], [8, 13], [1, 7],
                            [13, 9], [2, 6], [15, 6], [7, 2], [7, 11], [6, 15], [12, 10], [3, 7], [14, 7],
                            [4, 6], [9, 2], [5, 13], [8, 6], [10, 3], [8, 15], [10, 12], [1, 9], [13, 11],
                            [2, 8], [7, 4], [6, 8], [3, 0], [3, 9], [5, 6], [4, 8], [14, 9], [10, 5], [1, 2],
                            [0, 4], [10, 14], [11, 13], [2, 10], [6, 10], [12, 5], [4, 1], [12, 14], [14, 11],
                            [4, 10], [8, 1], [8, 10], [10, 7], [1, 4], [0, 6], [2, 3], [11, 6], [11, 15],
                            [6, 3], [6, 12], [12, 7], [3, 4], [8, 3], [10, 0], [8, 12], [1, 6], [0, 8],
                            [2, 5], [11, 8], [9, 11], [15, 8], [6, 14], [7, 13], [12, 9], [14, 6], [4, 5], [8, 5],
                            [10, 2], [9, 4], [0, 1], [5, 15], [8, 14], [10, 11], [9, 13], [0, 10], [2, 7],
                            [15, 10], [6, 7], [7, 15], [4, 7], [5, 8], [10, 4], [9, 6], [0, 3], [2, 0],
                            [10, 13], [9, 15], [11, 12], [2, 9], [13, 6], [6, 0], [13, 15], [15, 12], [6, 9],
                            [7, 8]])

        assert expected_edge_index == lifted_data.edge_index.to_dense(), "Something is wrong with the edge index."
