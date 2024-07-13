"""Test the message passing module."""

import networkx as nx
import torch
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
        lifted_dataset = self.lifting(self.data)

        # g = nx.Graph()
        # us, vs = lifted_dataset["edge_index"]

        # for u, v in zip(us, vs):
        #     g.add_edge(u, v)

        # nx.cycles.find_cycle(g)
