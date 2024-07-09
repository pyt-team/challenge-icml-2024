import pytest
import torch
import torch_geometric

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2hypergraph.mapper_lifting import (
    MapperCover,
    MapperLifting,
)

expected_edge_incidence = torch.tensor(
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


def naive_filter(data, filter):
    filter_dict = {
        "laplacian": Compose(
            [ToUndirected(), AddLaplacianEigenvectorPE(k=1, is_undirected=True)]
        ),
        "svd": SVDFeatureReduction(out_channels=1),
        "feature_sum": lambda data: torch.sum(data.x, dim=1).unsqueeze(1),
        "position_sum": lambda data: torch.sum(data.pos, dim=1).unsqueeze(1),
        "feature_pca": lambda data: torch.matmul(
            data.x, torch.pca_lowrank(data.x, q=1)[2][:, :1]
        ),
        "position_pca": lambda data: torch.matmul(
            data.pos, torch.pca_lowrank(data.pos, q=1)[2][:, :1]
        ),
    }
    transform = filter_dict[filter]
    filtered_data = transform(data)
    if filter == "laplacian":
        filtered_data = filtered_data["laplacian_eigenvector_pe"]
    elif name == "svd":
        filtered_data = filtered_data.x
    return filtered_data


"""Construct a cover_mask from filtered data and default lift parameters."""


def naive_cover(filtered_data):
    cover_mask = torch.full((filtered_data.shape[0], 10), False, dtype=torch.bool)
    data_min = torch.min(filtered_data)
    data_max = torch.max(filtered_data)
    data_range = torch.max(filtered_data) - torch.min(filtered_data)
    # width of each interval in the cover
    cover_width = data_range / (10 - (10 - 1) * 0.3)
    last = data_min + (10 - 1) * (1 - 0.3) * cover_width
    lows = torch.zeros(10)
    for i in range(10):
        lows[i] = (data_min) + (i) * (1 - 0.3) * cover_width
    highs = lows + cover_width
    for j, pt in enumerate(filtered_data):
        cover_mask[j] = (pt > lows) and (pt < highs)
    return cover_mask


class TestMapperLifting:
    "Test the MapperLifting class"

    def setup(self, filter):
        self.data = enriched_manual_graph()
        self.filter_name = filter
        self.mapper_lift = MapperLifting(filter_attr=filter)

    @pytest.mark.parametrize(
        "filter",
        [
            "laplacian",
            "svd",
            "feature_pca",
            "position_pca",
            "feature_sum",
            "position_sum",
        ],
    )
    def test_filter(self, filter):
        self.setup(filter)
        expected_filter_values = {
            "laplacian": torch.tensor(
                [
                    [0.3371],
                    [0.3611],
                    [0.0463],
                    [-0.4241],
                    [0.3611],
                    [-0.3546],
                    [-0.5636],
                    [-0.0158],
                ]
            ),
            "svd": torch.tensor(
                [
                    [-1.1183e00],
                    [-5.5902e00],
                    [-1.1180e01],
                    [-5.5902e01],
                    [-1.1180e02],
                    [-5.5902e02],
                    [-1.1180e03],
                    [-5.5902e03],
                ]
            ),
            "feature_pca": torch.tensor(
                [
                    [-1.1180e00],
                    [-5.5902e00],
                    [-1.1180e01],
                    [-5.5902e01],
                    [-1.1180e02],
                    [-5.5902e02],
                    [-1.1180e03],
                    [-5.5902e03],
                ]
            ),
            "position_pca": torch.tensor(
                [
                    [-0.7071],
                    [-3.5355],
                    [-6.3640],
                    [-9.1924],
                    [-12.0208],
                    [-14.8492],
                    [-17.6777],
                    [-20.5061],
                ]
            ),
            "feature_sum": torch.tensor(
                [
                    [5.0000e-01],
                    [2.5000e00],
                    [5.0000e00],
                    [2.5000e01],
                    [5.0000e01],
                    [2.5000e02],
                    [5.0000e02],
                    [2.5000e03],
                ]
            ),
            "position_sum": torch.tensor(
                [[1.0], [5.0], [9.0], [13.0], [17.0], [21.0], [25.0], [29.0]]
            ),
        }
        lift_filter_data = self.mapper_lift._filter(self.data)
        naive_filter_data = naive_filter(self.data, filter)
        assert naive_filter_data == lift_filter_data
        # assert torch.all(torch.isclose(expected_filter_values[self.filter_name],lift_filter_data)),\
        # f"Something is wrong with filtered values using {self.filter_name}.{lift_filter_data-expected_filter_values[self.filter_name]}."

    # def test_cover(self):
    #     # expected_cover_mask = {
    #     #     "laplacian": ,
    #     #     "svd": ,
    #     #     "feature_pca": ,
    #     #     "position_pca": ,
    #     #     "feature_sum": ,
    #     #     "position_sum": ,
    #     # }
    #     # expected_cover_mask = naive_cover(
    #     lift_cover_mask = self.mapper_lift.forward(self.data.clone()).cover
    #     assert expected_cover_mask[self.filter_name] == lift_cover_mask,\
    #     f"Something is wrong with the cover mask using {self.filter_name}."

    # def test_cluster(self):
    #     # expected_clusters = {
    #     #     "laplacian": ,
    #     #     "svd": ,
    #     #     "feature_pca": ,
    #     #     "position_pca": ,
    #     #     "feature_sum": ,
    #     #     "position_sum": ,
    #     # }
    #     lift_clusters = self.mapper_lift.forward(self.data.clone()).clusters
    #     assert expected_clusters[self.filter_name] == lift_clusters,\
    #     f"Something is wrong with the clustering using {self.filter_name}."

    # def test_lift_topology(self):
    #     # expected_hyperedge_incidence = {
    #     #     "laplacian": ,
    #     #     "svd": ,
    #     #     "feature_pca": ,
    #     #     "position_pca": ,
    #     #     "feature_sum": ,
    #     #     "position_sum": ,
    #     # }
    #     expected_incidence_1 = torch.cat(
    #         (expected_edge_incidence, expected_hyperedge_incidence[self.filter_name]),
    #         1
    #     )## MAYBE CHANGE DIMENSION!!!!!!!!!!!!!!!!!!!!!!!!!
    #     lifted_mapper = self.mapper_lift.forward(self.data.clone())
    #     assert (expected_incidence_1 == lifted_mapper.incidence_hyperedges.to_dense()).all(),\
    #     f"Something is wrong with the incidence hyperedges for the mapper lifting with {self.fitler_name}."

    #     assert expected_n_hyperedges == lifted_mapper.num_hyperedges,\
    #     f"Something is wrong with the number of hyperedges for the mapper lifting with {self.filter_name}."
