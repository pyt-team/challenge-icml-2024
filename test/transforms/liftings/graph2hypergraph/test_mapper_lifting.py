import torch
import torch_geometric

from modules.data.utils.utils import load_manual_graph
from modules.transforms.lifting.graph2hypergraph.mapper_lifting import (
    MapperCover,
    MapperLifting,
)

expected_edge_incidence = tensor(
    [
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
        ],
    ]
)


def enriched_manual_graph():
    data = load_manual_graph()
    undirected_edges = torch_geometric.utils.to_undirected(data.edge_index)
    new_x = torch.t(
        torch.tensor(
            [
                [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0],
                [-0.5, -2.5, -5.0, -25.0, -50.0, -250.0, -500.0, -2500.0],
            ]
        )
    )
    data.edge_index = undirected_edges
    data.x = new_x
    new_pos = torch.t(
        torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]])
    ).float()
    data.pos = new_pos
    return data


class TestMapperLifting:
    "Test the MapperLifting class"

    @pytest.mark.parametrize(
        "filter_name",
        [
            "laplacian",
            "svd",
            "feature_pca",
            "position_pca",
            "feature_sum",
            "position_sum",
        ],
    )
    def setup_method(self, filter_name):
        # Load the graph
        self.data = enriched_manual_graph()
        # Initialize the MapperLifting class
        self.filter_name = filter_name
        self.mapper_lift = MapperLifting(filter_attr=filter_name)

    def test_filter(self, filter_name):
        # expected_filter_values = {
        #     "laplacian": ,
        #     "svd": ,
        #     "feature_pca": ,
        #     "position_pca": ,
        #     "feature_sum": ,
        #     "position_sum": ,
        # }

        return None

    def test_cover(self):
        return None

    def test_cluster(self):
        return None

    def test_lift_topology(self):
        return None
