"""Test the message passing module."""

import networkx as nx
import torch
import torch_geometric

from modules.transforms.liftings.graph2simplicial.directed_clique_lifting import (
    DirectedSimplicialCliqueLifting,
)


class TestDirectedSimplicialCliqueLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_triangle_graph(self):
        """Sets up a test graph with 0 as a source node which
        generates a 2-simplex amongst (0,1,2), equivalent to
        the standard/undirected clique complex.
        """
        edges = [
            [0, 1],
            [0, 2],
            [2, 1],
        ]
        g = nx.DiGraph()
        g.add_edges_from(edges)
        edge_list = torch.Tensor(list(g.edges())).T.long()
        # Generate feature from 0 to 3
        x = torch.tensor([1, 5, 10]).unsqueeze(1).float()
        self.triangle_data = torch_geometric.data.Data(
            x=x, edge_index=edge_list, num_nodes=len(g.nodes)
        )

    def setup_three_two_graph(self):
        """Sets up a test graph with a single source node (0)
        with three edges emanating from it, and two sinks (1,2).

        The directed clique complex should result in two 2-simplices
        (0,2,3) and (0,1,3).
        """
        edges = [
            [0, 3],
            [0, 2],
            [0, 1],
            [3, 2],
            [3, 1],
        ]
        g = nx.DiGraph()
        g.add_edges_from(edges)
        edge_list = torch.Tensor(list(g.edges())).T.long()
        # Generate feature from 0 to 3
        x = torch.tensor([1, 5, 10, 50]).unsqueeze(1).float()
        self.three_two_data = torch_geometric.data.Data(
            x=x, edge_index=edge_list, num_nodes=len(g.nodes)
        )

    def setup_missing_triangle_graph(self):
        """Sets up a test graph with one clique with a single source
        and sink (0,1,3) and one without either (1,2,3).

        The directed clique complex should result in only one 2-simplex
        (0,1,3), the other clique is empty, illustrating the difference
        between the directed clique complex and the undirected clique
        complex.
        """
        edges = [
            [0, 3],
            [0, 1],
            [3, 2],
            [2, 1],
            [1, 3],
        ]
        g = nx.DiGraph()
        g.add_edges_from(edges)
        edge_list = torch.Tensor(list(g.edges())).T.long()
        # Generate feature from 0 to 3
        x = torch.tensor([1, 5, 10, 50]).unsqueeze(1).float()
        self.missing_triangle_data = torch_geometric.data.Data(
            x=x, edge_index=edge_list, num_nodes=len(g.nodes)
        )

    def setup_method(self):
        self.setup_triangle_graph()
        self.triangle_lifting = DirectedSimplicialCliqueLifting(complex_dim=2)

        self.setup_three_two_graph()
        self.three_two_lifting = DirectedSimplicialCliqueLifting(complex_dim=2)

        self.setup_missing_triangle_graph()
        self.missing_triangle_lifting = DirectedSimplicialCliqueLifting(complex_dim=2)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the triangle_graph lifting
        triangle_lifted_data = self.triangle_lifting.forward(self.triangle_data.clone())

        expected_triangle_incidence_1 = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        )

        assert (
            expected_triangle_incidence_1 == triangle_lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with triangle incidence_1 (nodes to edges)."

        # single triangle with all edges connected
        expected_triangle_incidence_2 = torch.tensor([[1.0], [1.0], [1.0]])

        assert (
            expected_triangle_incidence_2 == triangle_lifted_data.incidence_2.to_dense()
        ).all(), "Something is wrong with triangle incidence_2 (edges to triangles)."

        # Test the three_two_graph lifting
        three_two_lifted_data = self.three_two_lifting.forward(
            self.three_two_data.clone()
        )

        expected_three_two_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        assert (
            expected_three_two_incidence_1
            == three_two_lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with three_two incidence_1 (nodes to edges)."

        # five edges incident to two triangles, and the edge
        # connecting (0,3) is shared by both triangles.
        expected_three_two_incidence_2 = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        )

        assert (
            expected_three_two_incidence_2
            == three_two_lifted_data.incidence_2.to_dense()
        ).all(), "Something is wrong with three_two incidence_2 (edges to triangles)."

        # Test missing_triangle lifting
        missing_triangle_lifted_data = self.missing_triangle_lifting.forward(
            self.missing_triangle_data.clone()
        )

        expected_missing_triangle_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
            ]
        )

        assert (
            expected_missing_triangle_incidence_1
            == missing_triangle_lifted_data.incidence_1.to_dense()
        ).all(), (
            "Something is wrong with missing_triangle incidence_1 (nodes to edges)."
        )

        # only one triangle with the edges (3,2) and (2,1) ignored.
        expected_missing_triangle_incidence_2 = torch.tensor(
            [[1.0], [1.0], [0.0], [1.0], [0.0]]
        )

        assert (
            expected_missing_triangle_incidence_2
            == missing_triangle_lifted_data.incidence_2.to_dense()
        ).all(), (
            "Something is wrong with missing_triangle incidence_2 (edges to triangles)."
        )
