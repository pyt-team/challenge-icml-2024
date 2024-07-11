"""Test the message passing module."""


from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.pointcloud2simplicial.random_flag_complex import (
    RandomFlagComplexLifting,
)


class TestRandomFlagComplexLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = load_manual_graph()
        del self.data["edge_attr"]
        del self.data["edge_index"]

        self.lifting_p_0 = RandomFlagComplexLifting(steps=10, p=0)
        self.lifting_p_1 = RandomFlagComplexLifting(steps=1, p=1)
        self.lifting_hp = RandomFlagComplexLifting(steps=100, alpha=0.01)


    def test_empty(self):
        lifted_data = self.lifting_p_0.forward(self.data.clone())
        assert(lifted_data.x_1.size(0) == 0)

    def test_not_empty(self):
        lifted_data = self.lifting_hp.forward(self.data.clone())
        assert(lifted_data.x_1.size(0) > 0)

    def test_full_graph(self):
        lifted_data = self.lifting_p_1.forward(self.data.clone())
        possible_edges = lifted_data.num_nodes * (self.data.num_nodes - 1) / 2
        assert(lifted_data.x_1.size(0) == possible_edges)


        assert(lifted_data)
