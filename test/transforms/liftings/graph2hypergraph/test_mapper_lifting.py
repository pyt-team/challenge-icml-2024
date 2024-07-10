import pytest
import torch
import torch_geometric
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    Compose,
    SVDFeatureReduction,
    ToUndirected,
)

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


"""Construct a naive implementation to create the filtered data set given data and filter function."""


def naive_filter(data, filter):
    n_samples = data.x.shape[0]
    if filter == "laplacian":
        transform1 = ToUndirected()
        transform2 = AddLaplacianEigenvectorPE(k=1, is_undirected=True)
        filtered_data = transform2(transform1(data))
        filtered_data = filtered_data["laplacian_eigenvector_pe"]
    elif filter == "svd":
        svd = SVDFeatureReduction(out_channels=1)
        filtered_data = svd(data).x
    elif filter == "feature_sum":
        filtered_data = torch.zeros([n_samples, 1])
        for i in range(n_samples):
            for j in range(data.x.shape[1]):
                filtered_data[i] += data.x[i, j]
    elif filter == "position_sum":
        filtered_data = torch.zeros([n_samples, 1])
        for i in range(n_samples):
            for j in range(data.pos.shape[1]):
                filtered_data[i] += data.pos[i, j]
    elif filter == "feature_pca":
        U, S, V = torch.pca_lowrank(data.x, q=1)
        filtered_data = torch.matmul(data.x, V[:, :1])
    elif filter == "position_pca":
        U, S, V = torch.pca_lowrank(data.pos, q=1)
        filtered_data = torch.matmul(data.pos, V[:, :1])
    return filtered_data


"""Construct a cover_mask from filtered data and default lift parameters."""


def naive_cover(filtered_data):
    cover_mask = torch.full((filtered_data.shape[0], 10), False, dtype=torch.bool)
    data_min = torch.min(filtered_data) - 1e-3
    data_max = torch.max(filtered_data) + 1e-3
    data_range = data_max - data_min
    # width of each interval in the cover
    cover_width = data_range / (10 - (10 - 1) * 0.3)
    last = data_min + (10 - 1) * (1 - 0.3) * cover_width
    lows = torch.zeros(10)
    for i in range(10):
        lows[i] = (data_min) + (i) * (1 - 0.3) * cover_width
    highs = lows + cover_width
    # construct boolean cover
    for j, pt in enumerate(filtered_data):
        for i in range(10):
            if (pt > lows[i] or torch.isclose(pt, lows[i])) and (
                pt < highs[i] or torch.isclose(pt, highs[i])
            ):
                cover_mask[j, i] = True
    # delete empty covers
    keep = torch.full([10], True, dtype=torch.bool)
    count_falses = 0
    for i in range(10):
        for j in range(filtered_data.shape[0]):
            if not cover_mask[j, i]:
                count_falses += 1
        if count_falses == filtered_data.shape[0]:
            keep[i] = False
        count_falses = 0
    return torch.t(torch.t(cover_mask)[keep])


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
        lift_filter_data = self.mapper_lift._filter(self.data)
        naive_filter_data = naive_filter(self.data, filter)
        if filter != "laplacian":
            assert torch.all(
                torch.isclose(lift_filter_data, naive_filter_data)
            ), f"Something is wrong with filtered values using {self.filter_name}. The lifted filter data is {lift_filter_data} and the naive filter data is {naive_filter_data}."
        if filter == "laplacian":
            # laplacian produce eigenvector up to a unit multiple.
            # instead we check their absolute values.
            assert torch.all(
                torch.isclose(torch.abs(lift_filter_data), torch.abs(naive_filter_data))
            ), f"Something is wrong with filtered values using {self.filter_name}. The lifted filter data is {lift_filter_data} and the naive filter data is {naive_filter_data}."

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
    def test_cover(self, filter):
        self.setup(filter)
        transformed_data = self.mapper_lift.forward(self.data.clone())
        lift_cover_mask = self.mapper_lift.cover
        naive_cover_mask = naive_cover(self.mapper_lift.filtered_data[filter])
        assert torch.all(
            naive_cover_mask == lift_cover_mask
        ), f"Something is wrong with the cover mask using {self.filter_name}. Lifted cover mask is {lift_cover_mask} and naive cover mask {naive_cover_mask}."

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
    def test_cluster(self, filter):
        expected_clusters = {
            "laplacian": {
                0: (0, torch.tensor([6.0])),
                1: (1, torch.tensor([3.0])),
                2: (1, torch.tensor([5.0])),
                3: (2, torch.tensor([5.0])),
                4: (3, torch.tensor([7.0])),
                5: (4, torch.tensor([2.0, 7.0])),
                6: (5, torch.tensor([0.0, 1.0, 4.0])),
            },
            "svd": {
                0: (0, torch.tensor([7.0])),
                1: (1, torch.tensor([6.0])),
                2: (2, torch.tensor([5.0, 6.0])),
                3: (3, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
            },
            "feature_pca": {
                0: (0, torch.tensor([7.0])),
                1: (1, torch.tensor([6.0])),
                2: (2, torch.tensor([5.0, 6.0])),
                3: (3, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
            },
            "position_pca": {
                0: (0, torch.tensor([7.0])),
                1: (1, torch.tensor([6.0])),
                2: (2, torch.tensor([5.0])),
                3: (3, torch.tensor([4.0])),
                4: (4, torch.tensor([3.0])),
                5: (5, torch.tensor([2.0])),
                6: (6, torch.tensor([1.0])),
                7: (7, torch.tensor([0.0])),
            },
            "feature_sum": {
                0: (0, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
                1: (1, torch.tensor([5.0, 6.0])),
                2: (2, torch.tensor([6.0])),
                3: (3, torch.tensor([7.0])),
            },
            "position_sum": {
                0: (0, torch.tensor([0.0])),
                1: (1, torch.tensor([1.0])),
                2: (2, torch.tensor([2.0])),
                3: (3, torch.tensor([3.0])),
                4: (4, torch.tensor([4.0])),
                5: (5, torch.tensor([5.0])),
                6: (6, torch.tensor([6.0])),
                7: (7, torch.tensor([7.0])),
            },
        }
        self.setup(filter)
        transformed_data = self.mapper_lift.forward(self.data.clone())
        lift_clusters = self.mapper_lift.clusters
        if filter != "laplacian":
            assert (
                expected_clusters[self.filter_name].keys() == lift_clusters.keys()
            ), f"Different number of clusters using {filter}. Expected {list(expected_clusters[filter])} but got {list(lift_clusters)}."
            for cluster in lift_clusters.keys():
                assert (
                    expected_clusters[self.filter_name][cluster][0]
                    == lift_clusters[cluster][0]
                )
                assert torch.equal(
                    expected_clusters[self.filter_name][cluster][1],
                    lift_clusters[cluster][1],
                ), f"Something is wrong with the clustering using {self.filter_name}. Expected node subset {expected_clusters[self.filter_name][cluster][1]} but got {lift_clusters[cluster][1]} for cluster {cluster}."
        # Laplacian function projects up to a unit. This causes clusters to not be identical
        # instead we check if the node subsets of the lifted set are somewhere in the expected set.
        if filter == "laplacian":
            assert len(lift_clusters) == len(
                expected_clusters["laplacian"]
            ), f"Different number of clusters using {filter}. Expected {len(expected_clusters[filter])} clusters but got {len(lift_clusters)}."
            lift_cluster_nodes = [value[1].tolist() for value in lift_clusters.values()]
            expected_cluster_nodes = [
                value[1].tolist() for value in expected_clusters[filter].values()
            ]
            for node_subset in lift_cluster_nodes:
                assert (
                    node_subset in expected_cluster_nodes
                ), f"{node_subset} is a cluster not in {expected_cluster_nodes} but in {lift_cluster_nodes}."
                expected_cluster_nodes.remove(node_subset)
            assert (
                expected_cluster_nodes == []
            ), f"Expected clusters contain more clusters than in the lifted cluster."

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
    def test_lift_topology(self, filter):
        expected_lift = {
            "laplacian1": {
                "num_hyperedges": 33,
                "hyperedge_incidence": torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
            "laplacian2": {
                "num_hyperedges": 33,
                "hyperedge_incidence": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    ]
                ),
            },
            "svd": {
                "num_hyperedges": 30,
                "hyperedge_incidence": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
            "feature_pca": {
                "num_hyperedges": 30,
                "hyperedge_incidence": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
            "position_pca": {
                "num_hyperedges": 34,
                "hyperedge_incidence": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
            "feature_sum": {
                "num_hyperedges": 30,
                "hyperedge_incidence": torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            "position_sum": {
                "num_hyperedges": 34,
                "hyperedge_incidence": torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
        }
        self.setup(filter)
        lifted_mapper = self.mapper_lift.forward(self.data.clone())
        if filter != "laplacian":
            expected_n_hyperedges = expected_lift[self.filter_name]["num_hyperedges"]
            expected_incidence_1 = torch.hstack(
                [
                    expected_edge_incidence,
                    expected_lift[self.filter_name]["hyperedge_incidence"],
                ]
            )
            assert (
                expected_incidence_1 == lifted_mapper.incidence_hyperedges.to_dense()
            ).all(), f"Something is wrong with the incidence hyperedges for the mapper lifting with {self.filter_name}."
        if filter == "laplacian":
            expected_n_hyperedges1 = expected_lift["laplacian1"]["num_hyperedges"]
            expected_n_hyperedges2 = expected_lift["laplacian2"]["num_hyperedges"]
            assert expected_n_hyperedges1 == expected_n_hyperedges2
            expected_n_hyperedges = expected_n_hyperedges1
            expected_incidence_11 = torch.hstack(
                [
                    expected_edge_incidence,
                    expected_lift["laplacian1"]["hyperedge_incidence"],
                ]
            )
            expected_incidence_12 = torch.hstack(
                [
                    expected_edge_incidence,
                    expected_lift["laplacian2"]["hyperedge_incidence"],
                ]
            )
            assert (
                expected_incidence_11 == lifted_mapper.incidence_hyperedges.to_dense()
            ).all() or (
                expected_incidence_12 == lifted_mapper.incidence_hyperedges.to_dense()
            ).all(), f"Something is wrong with the incidence hyperedges for the mapper lifting with {self.filter_name}. lifted incidence is {lifted_mapper.incidence_hyperedges.to_dense()}"

        assert (
            expected_n_hyperedges == lifted_mapper.num_hyperedges
        ), f"Something is wrong with the number of hyperedges for the mapper lifting with {self.filter_name}."
