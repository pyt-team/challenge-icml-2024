"""Test the cell encoding module."""

import networkx as nx
from torch_geometric.utils import to_networkx

from modules.data.utils.utils import load_manual_cell_complex
from modules.transforms.liftings.cell2graph.cell_encoding_lifting import (
    CellEncodingLifting,
)


class TestCellEncodingLifting:
    """Test the CellEncoding class."""

    def setup_method(self):
        # Load the cell complex
        self.data = load_manual_cell_complex()

        # Initialize the CellEncodingLifting class
        self.lifting = CellEncodingLifting()

    def test_lift_topology(self):
        # test the lift topology method
        lifted_data = self.lifting.forward(self.data.clone())
        lifted_graph = to_networkx(lifted_data, node_attrs=["x"], to_undirected=True)

        expected_graph = nx.Graph()
        expected_graph.add_nodes_from(range(8))
        expected_graph.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (0, 4),
                (0, 5),
                (1, 2),
                (1, 4),
                (1, 6),
                (2, 3),
                (2, 5),
                (2, 6),
                (2, 7),
                (3, 7),
                (4, 5),
                (4, 6),
                (4, 8),
                (5, 6),
                (5, 8),
                (6, 8),
            ]
        )

        assert nx.is_isomorphic(
            lifted_graph, expected_graph
        ), "Something in the lifted graph structure is wrong."

        expected_node_features = {
            0: [1.0, 0.0, 0.0],
            1: [1.0, 0.0, 0.0],
            2: [1.0, 0.0, 0.0],
            3: [1.0, 0.0, 0.0],
            4: [0.0, 1.0, 0.0],
            5: [0.0, 1.0, 0.0],
            6: [0.0, 1.0, 0.0],
            7: [0.0, 1.0, 0.0],
            8: [0.0, 0.0, 1.0],
        }

        assert (
            nx.get_node_attributes(lifted_graph, "x") == expected_node_features
        ), "Something in the lifted graph structure is wrong."
