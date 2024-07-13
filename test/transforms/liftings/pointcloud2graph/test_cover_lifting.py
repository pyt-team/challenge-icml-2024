"""Test the message passing module."""

import networkx as nx
from torch_geometric.utils.convert import to_networkx

from modules.data.utils.utils import load_annulus
from modules.transforms.liftings.pointcloud2graph.cover_lifting import CoverLifting


class TestCoverLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Load the point cloud
        self.data = load_annulus()

        # Initialise the CoverLifting class
        self.lifting = CoverLifting()

    def test_lift_topology(self):
        """Test the lift_topology method."""
        # Test the lift_topology method
        lifted_data = self.lifting(self.data)

        g = to_networkx(lifted_data, to_undirected=True)
        nx.find_cycle(g)
