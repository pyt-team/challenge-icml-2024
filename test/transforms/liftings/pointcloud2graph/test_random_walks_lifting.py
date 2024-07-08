import numpy as np
from torch_geometric.nn import knn_graph
from modules.data.utils.utils import load_random_point_cloud
from modules.transforms.liftings.pointcloud2graph.random_walks_lifting import GraphRandomWalksLifting

class TestGraphRandomWalksLifting:
    """Test the GraphRandomWalksLifting class."""

    def setup_method(self):
        """Set up the test method."""
        self.data = load_random_point_cloud()

        # Initialise the GraphRandomWalksLifting class
        self.lifting = GraphRandomWalksLifting(k=3, num_walks=10, steps_per_walk=2)

    def test_parameter_set(self):
        """Test if parameters are set correctly."""
        assert self.lifting.k == 3, "Parameter k is not set correctly"
        assert self.lifting.num_walks == 10, "Parameter num_walks is not set correctly"
        assert self.lifting.steps_per_walk == 2, "Parameter steps_per_walk is not set correctly"

    def test_calculate_edge_weights(self):
        """Test the edge weight calculation."""
        point_cloud = self.data.pos.numpy()
        edge_index = knn_graph(self.data.pos, k=self.lifting.k)

        edge_weights = self.lifting._calculate_edge_weights(point_cloud, edge_index)

        assert len(edge_weights) == edge_index.size(1), "Edge weights length mismatch"
        assert all(weight >= 0 for weight in edge_weights), "Negative edge weights found"

    def test_create_weighted_networkx_graph(self):
        """Test the creation of weighted NetworkX graph."""
        point_cloud = self.data.pos.numpy()
        edge_index = knn_graph(self.data.pos, k=self.lifting.k)

        edge_weights = self.lifting._calculate_edge_weights(point_cloud, edge_index)
        G = self.lifting._create_weighted_networkx_graph(edge_index, edge_weights)

        assert G.number_of_nodes() == self.data.num_nodes, "Number of nodes mismatch"

    def test_normalize_edge_weights(self):
        """Test the normalization of edge weights."""
        point_cloud = self.data.pos.numpy()
        edge_index = knn_graph(self.data.pos, k=self.lifting.k)

        edge_weights = self.lifting._calculate_edge_weights(point_cloud, edge_index)
        G = self.lifting._create_weighted_networkx_graph(edge_index, edge_weights)
        self.lifting._normalize_edge_weights(G)

        for node in G.nodes():
            total_prob = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
            assert np.isclose(total_prob, 1.0), f"Total probability for node {node} is not 1.0"

    def test_random_walk(self):
        """Test the random walk functionality."""
        point_cloud = self.data.pos.numpy()
        edge_index = knn_graph(self.data.pos, k=self.lifting.k)

        edge_weights = self.lifting._calculate_edge_weights(point_cloud, edge_index)
        G = self.lifting._create_weighted_networkx_graph(edge_index, edge_weights)
        self.lifting._normalize_edge_weights(G)

        start_node = 0
        steps = 10
        path = self.lifting._random_walk(G, start_node, steps)

        assert len(path) == steps + 1, "Path length mismatch"
        assert path[0] == start_node, "Random walk did not start at the correct node"

    def test_create_topological_graph(self):
        """Test the creation of the topological graph."""
        point_cloud = self.data.pos.numpy()
        edge_index = knn_graph(self.data.pos, k=self.lifting.k)

        edge_weights = self.lifting._calculate_edge_weights(point_cloud, edge_index)
        G = self.lifting._create_weighted_networkx_graph(edge_index, edge_weights)
        self.lifting._normalize_edge_weights(G)

        num_walks = self.lifting.num_walks
        steps_per_walk = self.lifting.steps_per_walk
        topological_graph = self.lifting._create_topological_graph(G, num_walks, steps_per_walk)

        assert topological_graph.number_of_nodes() == self.data.num_nodes, "Number of nodes in topological graph mismatch"
        assert topological_graph.number_of_edges() > 0, "No edges in topological graph"

    def test_lift_topology(self):
        """Test the lift topology functionality."""
        result = self.lifting.lift_topology(self.data)

        assert 'num_nodes' in result, "Result does not contain 'num_nodes'"
        assert 'edge_index' in result, "Result does not contain 'edge_index'"
        assert result['num_nodes'] == self.data.num_nodes, "Number of nodes mismatch in lifted topology"
        assert result['edge_index'].shape[0] == 2, "Edge index shape mismatch"
