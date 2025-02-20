import hashlib
import itertools as it
import os
import os.path as osp
import pickle
import tempfile
import zipfile
from collections.abc import Callable
from urllib.request import urlretrieve

import networkx as nx
import numpy as np
import omegaconf
import rootutils
import toponetx.datasets.graph as graph
import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.transforms as T
import torch_sparse
from gudhi.datasets.generators import points
from gudhi.datasets.remote import (
    fetch_bunny,
    fetch_daily_activities,
    fetch_spiral_2d,
)
from topomodelx.utils.sparse import from_sparse
from torch_geometric.data import Data
from torch_geometric.datasets import GeometricShapes
from torch_sparse import SparseTensor, coalesce

rootutils.setup_root("./", indicator=".project-root", pythonpath=True)


def get_ccc_connectivity(complex, max_rank):
    r"""

    Parameters
    ----------
    complex : topnetx.CombinatorialComplex, topnetx.SimplicialComplex
        Combinatorial Complex complex.
    max_rank : int
        Maximum rank of the complex.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )

    connectivity = {}
    # compute incidence matrices
    for rank_idx in range(1, max_rank + 1):
        matrix = complex.incidence_matrix(rank=rank_idx - 1, to_rank=rank_idx)
        connectivity[f"incidence_{rank_idx}"] = from_sparse(matrix)

    # compute adjacent matrices
    for rank_idx in range(max_rank + 1):
        matrix = complex.adjacency_matrix(rank_idx, rank_idx + 1)
        connectivity[f"adjacency_{rank_idx}"] = from_sparse(matrix)

    for rank_idx in range(1, max_rank + 1):
        matrix = complex.laplacian_matrix(rank_idx)
        connectivity[f"laplacian_{rank_idx}"] = matrix

    connectivity["shape"] = practical_shape

    return connectivity


def get_complex_connectivity(complex, max_rank, signed=False):
    r"""Gets the connectivity matrices for the complex.

    Parameters
    ----------
    complex : topnetx.CellComplex, topnetx.SimplicialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    signed : bool
        If True, returns signed connectivity matrices.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "hodge_laplacian",
        ]:
            try:
                connectivity[f"{connectivity_info}_{rank_idx}"] = from_sparse(
                    getattr(complex, f"{connectivity_info}_matrix")(
                        rank=rank_idx, signed=signed
                    )
                )
            except ValueError:  # noqa: PERF203
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    connectivity["shape"] = practical_shape
    return connectivity


def get_combinatorial_complex_connectivity(complex, max_rank=None):
    r"""Gets the connectivity matrices for the combinatorial complex.

    Parameters
    ----------
    complex : topnetx.CombinatorialComplex
        Combinatorial complex.
    max_rank : int
        Maximum rank of the complex.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    if max_rank is None:
        max_rank = complex.dim
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )

    connectivity = {}

    for rank_idx in range(max_rank + 1):
        if rank_idx > 0:
            try:
                connectivity[f"incidence_{rank_idx}"] = from_sparse(
                    complex.incidence_matrix(
                        rank=rank_idx - 1, to_rank=rank_idx
                    )
                )
            except ValueError:
                connectivity[f"incidence_{rank_idx}"] = (
                    generate_zero_sparse_connectivity(
                        m=practical_shape[rank_idx],
                        n=practical_shape[rank_idx],
                    )
                )

        try:
            connectivity[f"adjacency_{rank_idx}"] = from_sparse(
                complex.adjacency_matrix(rank=rank_idx, via_rank=rank_idx + 1)
            )
        except ValueError:
            connectivity[f"adjacency_{rank_idx}"] = (
                generate_zero_sparse_connectivity(
                    m=practical_shape[rank_idx], n=practical_shape[rank_idx]
                )
            )

        connectivity["shape"] = practical_shape

    return connectivity


def generate_zero_sparse_connectivity(m, n):
    r"""Generates a zero sparse connectivity matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    torch.sparse_coo_tensor
        Zero sparse connectivity matrix.
    """
    return torch.sparse_coo_tensor((m, n)).coalesce()


def load_random_shape_point_cloud(seed=None, num_points=64, num_classes=2):
    """Create a toy point cloud dataset"""
    rng = np.random.default_rng(seed)
    dataset = GeometricShapes(root="data/GeometricShapes")
    dataset.transform = T.SamplePoints(num=num_points)
    data = dataset[rng.integers(40)]
    data.y = rng.integers(num_classes, size=num_points)
    data.x = torch.tensor(
        rng.integers(2, size=(num_points, 1)), dtype=torch.float
    )
    return data


def load_cell_complex_dataset(cfg):
    r"""Loads cell complex datasets."""


def load_simplicial_dataset(cfg):
    r"""Loads simplicial datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Simplicial dataset.
    """
    if cfg["data_name"] == "KarateClub":
        data = graph.karate_club(complex_type="simplicial", feat_dim=2)
        max_rank = data.dim
        features = {}
        dict_feat_equivalence = {
            0: "node_feat",
            1: "edge_feat",
            2: "face_feat",
            3: "tetrahedron_feat",
        }
        for rank_idx in range(max_rank + 1):
            try:
                features[f"x_{rank_idx}"] = torch.tensor(
                    np.stack(
                        list(
                            data.get_simplex_attributes(
                                dict_feat_equivalence[rank_idx]
                            ).values()
                        )
                    )
                )
            except ValueError:  # noqa: PERF203
                features[f"x_{rank_idx}"] = torch.tensor(
                    np.zeros((data.shape[rank_idx], 0))
                )
        features["y"] = torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        # features['num_nodes'] = data.shape[0]
        features["x"] = features["x_0"]
        connectivity = get_complex_connectivity(data, max_rank)
        data = torch_geometric.data.Data(**connectivity, **features)

        # Project node-level features to edge-level (WHY DO WE NEED IT, data already has x_1)
        data.x_1 = data.x_1 + torch.mm(data.incidence_1.to_dense().T, data.x_0)

        return torch_geometric.transforms.random_node_split.RandomNodeSplit(
            num_val=4, num_test=4
        )(data)

    if cfg["data_name"] == "wall_shear_stress":
        path_to_data_dir = osp.join(rootutils.find_root(), cfg["data_dir"])
        path_to_npz = osp.join(path_to_data_dir, f"{cfg['data_name']}.npz")

        # Download data
        if not osp.exists(path_to_npz):
            os.makedirs(path_to_data_dir, exist_ok=True)
            urlretrieve(
                "https://surfdrive.surf.nl/files/index.php/s/6h2MLfnxvQJLx1W/download",
                path_to_npz,
            )

        # Load data
        npz = np.load(path_to_npz)

        data = Data(**{key: torch.from_numpy(npz[key]) for key in npz.files})

        # Node attributes (geodesic distance to artery inlet)
        data.x = data.x.view(-1, 1)
        # data.x_0 = data.x

        # Face attributes (surface normal) (should this be "x_2"?)
        data.x_1 = torch.nn.functional.normalize(
            torch.cross(
                data.pos[data.face[1]] - data.pos[data.face[0]],
                data.pos[data.face[2]] - data.pos[data.face[0]],
                dim=1,
            )
        )

        # Incidence from nodes to faces (should this be "incidence_2"?)
        # Tried using TopoNetX for this but it crashed the Jupyter notebook
        num_face = data.face.size(1)
        data.incidence_1 = torch.sparse_coo_tensor(
            torch.stack(
                (
                    data.face.T.reshape(-1),
                    torch.arange(num_face).expand((3, -1)).T.reshape(-1),
                )
            ),
            torch.ones(3 * num_face),
        )

        # Up- and down-Laplacian, etc. should be computed here

        return data

    return NotImplementedError


def load_hypergraph_pickle_dataset(cfg):
    r"""Loads hypergraph datasets from pickle files.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Hypergraph dataset.
    """
    data_dir = cfg["data_dir"]
    print(f"Loading {cfg['data_domain']} dataset name: {cfg['data_name']}")

    # Load node features:

    with open(osp.join(data_dir, "features.pickle"), "rb") as f:
        features = pickle.load(f)
        features = features.todense()

    # Load node labels:
    with open(osp.join(data_dir, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # Load hypergraph.
    with open(osp.join(data_dir, "hypergraph.pickle"), "rb") as f:
        # Hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f"number of hyperedges: {len(hypergraph)}")

    node_list = []
    edge_list = []

    for edge_idx, cur_he in enumerate(hypergraph.values()):
        cur_size = len(cur_he)
        node_list.extend(cur_he)
        edge_list.extend([edge_idx] * cur_size)

    # check that every node is in some hyperedge
    if len(np.unique(node_list)) != num_nodes:
        # add self hyperedges to isolated nodes
        isolated_nodes = np.setdiff1d(
            np.arange(num_nodes), np.unique(node_list)
        )

        for node in isolated_nodes:
            node_list += [node]
            edge_list += [edge_idx]
            edge_idx += 1
            hypergraph[f"Unique_additonal_he_{edge_idx}"] = [node]

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(
        x=features,
        x_0=features,
        edge_index=edge_index,
        incidence_hyperedges=edge_index,
        y=labels,
    )

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    num_class = len(np.unique(labels.numpy()))

    # Add parameters to attribute
    data.n_x = n_x
    data.num_hyperedges = len(hypergraph)
    data.num_class = num_class

    data.incidence_hyperedges = torch.sparse_coo_tensor(
        data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(data.num_nodes, data.num_hyperedges),
    )

    # Print some info
    print("Final num_hyperedges", data.num_hyperedges)
    print("Final num_nodes", data.num_nodes)
    print("Final num_class", data.num_class)

    return data


def load_point_cloud(
    num_classes: int = 2, num_samples: int = 18, seed: int = 42
):
    """Create a toy point cloud dataset"""
    rng = np.random.default_rng(seed)

    points = torch.tensor(rng.random((num_samples, 2)), dtype=torch.float)
    classes = torch.tensor(
        rng.integers(num_classes, size=num_samples), dtype=torch.long
    )
    features = torch.tensor(
        rng.integers(3, size=(num_samples, 1)), dtype=torch.float
    )

    return torch_geometric.data.Data(x=features, y=classes, pos=points)


def load_manual_graph():
    """Create a manual graph for testing purposes."""
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the edges
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [5, 7],
        [2, 7],
        [0, 7],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    edge_list = torch.Tensor(list(G.edges())).T.long()
    edge_list = torch_geometric.utils.to_undirected(edge_list)

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )


def load_k4_graph() -> torch_geometric.data.Data:
    """K_4 is a complete graph with 4 vertices."""
    vertices = [i for i in range(4)]
    y = [0, 1, 1, 1]
    edges = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ]
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()
    x = torch.tensor([1, 5, 10, 50]).unsqueeze(1).float()
    return torch_geometric.data.Data(
        x=x, edge_index=edge_list, num_nodes=len(vertices), y=torch.tensor(y)
    )


def load_double_house_graph() -> torch_geometric.data.Data:
    """Double house graph is a featured graph in Geiger et al."""
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    edges = [
        [0, 1],
        [0, 2],
        [0, 7],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [3, 4],
        [4, 6],
        [5, 6],
        [5, 7],
        [6, 7],
    ]
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from([[v1, v2] for (v1, v2) in edges])
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()
    return torch_geometric.data.Data(
        x=x, edge_index=edge_list, num_nodes=len(vertices), y=torch.tensor(y)
    )


def load_8_vertex_cubic_graphs() -> list[torch_geometric.data.Data]:
    """Downloaded from https://mathrepo.mis.mpg.de/GraphCurveMatroids/"""
    # fmt: off
    edgesets = [
        [{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {5, 8}, {6, 7}, {6, 8}, {7, 8}],
        [{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 5}, {3, 6}, {4, 5}, {4, 7}, {5, 8}, {6, 7}, {6, 8}, {7, 8}],
        [{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 5}, {3, 6}, {4, 7}, {4, 8}, {5, 7}, {5, 8}, {6, 7}, {6, 8}],
        [{1, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6}, {3, 5}, {3, 7}, {4, 6}, {4, 7}, {5, 8}, {6, 8}, {7, 8}],
        [{1, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6}, {3, 5}, {3, 7}, {4, 6}, {4, 8}, {5, 8}, {6, 7}, {7, 8}],
    ]
    # fmt: on

    list_data = []
    for i, edgeset in enumerate(edgesets):
        n = 8 if i < 5 else 10
        vertices = [i for i in range(n)]
        x = (
            torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000])
            .unsqueeze(1)
            .float()
            if i < 5
            else torch.tensor(
                [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
            )
            .unsqueeze(1)
            .float()
        )
        y = (
            torch.tensor([0, 1, 1, 1, 0, 0, 0, 0])
            if i < 5
            else torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        )
        edgeset = [[v1 - 1, v2 - 1] for (v1, v2) in edgeset]
        G = nx.Graph()
        G.add_nodes_from(vertices)
        # offset by 1, since the graphs presented start at 1.
        G.add_edges_from(edgeset)
        G.to_undirected()
        edge_list = torch.Tensor(list(G.edges())).T.long()

        data = torch_geometric.data.Data(
            x=x, edge_index=edge_list, num_nodes=n, y=y
        )

        list_data.append(data)
    return list_data


def load_manual_mol():
    """Create a manual graph for testing the ring implementation.
    Actually is the 471 molecule of QM9 dataset."""
    # Define the vertices
    vertices = [i for i in range(12)]
    y = torch.tensor(
        [
            [
                2.2569e00,
                4.5920e01,
                -6.3076e00,
                1.9211e00,
                8.2287e00,
                4.6414e02,
                2.6121e00,
                -8.3351e03,
                -8.3349e03,
                -8.3349e03,
                -8.3359e03,
                2.0187e01,
                -4.8740e01,
                -4.9057e01,
                -4.9339e01,
                -4.5375e01,
                6.5000e00,
                3.8560e00,
                3.0122e00,
            ]
        ]
    )

    # Define the edges
    edges = [
        [0, 1],
        [0, 6],
        [1, 0],
        [1, 2],
        [1, 3],
        [1, 5],
        [2, 1],
        [2, 3],
        [2, 7],
        [2, 8],
        [3, 1],
        [3, 2],
        [3, 4],
        [3, 9],
        [4, 3],
        [4, 5],
        [5, 1],
        [5, 4],
        [5, 10],
        [5, 11],
        [6, 0],
        [7, 2],
        [8, 2],
        [9, 3],
        [10, 5],
        [11, 5],
    ]

    # Add smile
    smiles = "[H]O[C@@]12C([H])([H])O[C@]1([H])C2([H])[H]"

    # # Create a graph
    # G = nx.Graph()

    # # Add vertices
    # G.add_nodes_from(vertices)

    # # Add edges
    # G.add_edges_from(edges)

    # G.to_undirected()
    # edge_list = torch.Tensor(list(G.edges())).T.long()

    x = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    pos = torch.tensor(
        [
            [-0.0520, 1.4421, 0.0438],
            [-0.0146, 0.0641, 0.0278],
            [-0.2878, -0.7834, -1.1968],
            [-1.1365, -0.9394, 0.0399],
            [-0.4768, -1.7722, 0.9962],
            [0.6009, -0.8025, 1.1266],
            [0.6168, 1.7721, -0.5660],
            [-0.7693, -0.2348, -2.0014],
            [0.3816, -1.5834, -1.5029],
            [-2.2159, -0.8594, 0.0798],
            [1.5885, -1.2463, 0.9538],
            [0.5680, -0.3171, 2.1084],
        ]
    )

    assert len(x) == len(vertices)
    assert len(pos) == len(vertices)

    return torch_geometric.data.Data(
        x=torch.tensor(x).float(),
        edge_index=torch.tensor(edges).T.long(),
        num_nodes=len(vertices),
        y=torch.tensor(y),
        smiles=smiles,
        pos=pos,
    )


def load_manual_hypergraph():
    """Create a manual hypergraph for testing purposes."""
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the hyperedges
    hyperedges = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
        [3, 4],
        [4, 5],
        [4, 7],
        [5, 6],
        [6, 7],
    ]

    # Generate feature from 0 to 7
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()
    labels = torch.tensor(y, dtype=torch.long)

    node_list = []
    edge_list = []

    for edge_idx, he in enumerate(hyperedges):
        cur_size = len(he)
        node_list += he
        edge_list += [edge_idx] * cur_size

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    incidence_hyperedges = torch.sparse_coo_tensor(
        edge_index,
        values=torch.ones(edge_index.shape[1]),
        size=(len(vertices), len(hyperedges)),
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=labels,
        incidence_hyperedges=incidence_hyperedges,
    )


def load_manual_hypergraph_2(cfg: dict):
    """Create a manual hypergraph for testing purposes."""
    rng = np.random.default_rng(1234)
    n, m = 12, 24
    hyperedges = set(
        [tuple(np.flatnonzero(rng.choice([0, 1], size=n))) for _ in range(m)]
    )
    hyperedges = [np.array(he) for he in hyperedges]
    R = torch.tensor(np.concatenate(hyperedges), dtype=torch.long)
    C = torch.tensor(
        np.repeat(np.arange(len(hyperedges)), [len(he) for he in hyperedges]),
        dtype=torch.long,
    )
    V = torch.tensor(np.ones(len(R)))
    incidence_hyperedges = torch_sparse.SparseTensor(row=R, col=C, value=V)
    incidence_hyperedges = (
        incidence_hyperedges.coalesce().to_torch_sparse_coo_tensor()
    )

    ## Bipartite graph repr.
    edges = np.array(
        list(
            it.chain(
                *[[(i, v) for v in he] for i, he in enumerate(hyperedges)]
            )
        )
    )
    return Data(
        x=torch.empty((n, 0)),
        edge_index=torch.tensor(edges, dtype=torch.long),
        num_nodes=n,
        num_node_features=0,
        num_edges=len(hyperedges),
        incidence_hyperedges=incidence_hyperedges,
        max_dim=cfg.get("max_dim", 3),
    )


def load_contact_primary_school(cfg: dict, data_dir: str):
    import gdown

    url = "https://drive.google.com/uc?id=1H7PGDPvjCyxbogUqw17YgzMc_GHLjbZA"
    fn = tempfile.NamedTemporaryFile()  # noqa: SIM115, RUF100
    gdown.download(url, fn.name, quiet=False)
    archive = zipfile.ZipFile(fn.name, "r")
    labels = archive.open(
        "contact-primary-school/node-labels-contact-primary-school.txt", "r"
    ).readlines()
    hyperedges = archive.open(
        "contact-primary-school/hyperedges-contact-primary-school.txt", "r"
    ).readlines()
    label_names = archive.open(
        "contact-primary-school/label-names-contact-primary-school.txt", "r"
    ).readlines()

    hyperedges = [
        list(map(int, he.decode().replace("\n", "").strip().split(",")))
        for he in hyperedges
    ]
    labels = np.array(
        [int(b.decode().replace("\n", "").strip()) for b in labels]
    )
    label_names = np.array(
        [b.decode().replace("\n", "").strip() for b in label_names]
    )

    # Based on: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html
    HE_coo = torch.tensor(
        np.array(
            [
                np.concatenate(hyperedges),
                np.repeat(
                    np.arange(len(hyperedges)), [len(he) for he in hyperedges]
                ),
            ]
        )
    )

    incidence_hyperedges = (
        SparseTensor(
            row=HE_coo[0, :],
            col=HE_coo[1, :],
            value=torch.tensor(np.ones(HE_coo.shape[1])),
        )
        .coalesce()
        .to_torch_sparse_coo_tensor()
    )

    return Data(
        x=torch.empty((len(labels), 0)),
        edge_index=HE_coo,
        y=torch.LongTensor(labels),
        y_names=label_names,
        num_nodes=len(labels),
        num_node_features=0,
        num_edges=len(hyperedges),
        incidence_hyperedges=incidence_hyperedges,
        max_dim=cfg.get("max_dim", 1),
        # x_hyperedges=torch.tensor(np.empty(shape=(len(hyperedges), 0)))
    )


def load_senate_committee(
    cfg: dict, data_dir: str
) -> torch_geometric.data.Data:
    import tempfile
    import zipfile

    import gdown

    url = "https://drive.google.com/uc?id=17ZRVwki_x_C_DlOAea5dPBO7Q4SRTRRw"
    fn = tempfile.NamedTemporaryFile()  # noqa: SIM115, RUF100
    gdown.download(url, fn.name, quiet=False)
    archive = zipfile.ZipFile(fn.name, "r")
    labels = archive.open(
        "senate-committees/node-labels-senate-committees.txt", "r"
    ).readlines()
    hyperedges = archive.open(
        "senate-committees/hyperedges-senate-committees.txt", "r"
    ).readlines()
    label_names = archive.open(
        "senate-committees/node-names-senate-committees.txt", "r"
    ).readlines()

    hyperedges = [
        list(map(int, he.decode().replace("\n", "").strip().split(",")))
        for he in hyperedges
    ]
    labels = np.array(
        [int(b.decode().replace("\n", "").strip()) for b in labels]
    )
    label_names = np.array(
        [b.decode().replace("\n", "").strip() for b in label_names]
    )

    # Based on: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html
    HE_coo = torch.tensor(
        np.array(
            [
                np.concatenate(hyperedges) - 1,
                np.repeat(
                    np.arange(len(hyperedges)), [len(he) for he in hyperedges]
                ),
            ]
        )
    )
    from torch_sparse import SparseTensor

    incidence_hyperedges = (
        SparseTensor(
            row=HE_coo[0, :],
            col=HE_coo[1, :],
            value=torch.tensor(np.ones(HE_coo.shape[1])),
        )
        .coalesce()
        .to_torch_sparse_coo_tensor()
    )

    return Data(
        x=torch.empty((len(labels), 0)),
        edge_index=HE_coo,
        y=torch.LongTensor(labels),
        y_names=label_names,
        num_nodes=len(labels),
        num_node_features=0,
        num_edges=len(hyperedges),
        incidence_hyperedges=incidence_hyperedges,
        max_dim=cfg.get("max_dim", 2),
        # x_hyperedges=torch.tensor(np.empty(shape=(len(hyperedges), 0)))
    )


def get_Planetoid_pyg(cfg):
    r"""Loads Planetoid graph datasets from torch_geometric.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Graph dataset.
    """
    data_dir, data_name = cfg["data_dir"], cfg["data_name"]
    dataset = torch_geometric.datasets.Planetoid(data_dir, data_name)
    data = dataset.data
    data.num_nodes = data.x.shape[0]
    return data


def get_TUDataset_pyg(cfg):
    r"""Loads TU graph datasets from torch_geometric.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    list
        List containing the graph dataset.
    """
    data_dir, data_name = cfg["data_dir"], cfg["data_name"]
    dataset = torch_geometric.datasets.TUDataset(root=data_dir, name=data_name)
    return [data for data in dataset]


def ensure_serializable(obj):
    r"""Ensures that the object is serializable.

    Parameters
    ----------
    obj : object
        Object to ensure serializability.

    Returns
    -------
    object
        Object that is serializable.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = ensure_serializable(value)
        return obj
    elif isinstance(obj, list | tuple):  # noqa: RET505
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, str | int | float | bool | type(None)):
        return obj
    elif isinstance(obj, omegaconf.dictconfig.DictConfig):
        return dict(obj)
    else:
        return None


def make_hash(o):
    r"""Makes a hash from a dictionary, list, tuple or set to any level, that
    contains only other hashable types (including any lists, tuples, sets, and
    dictionaries).

    Parameters
    ----------
    o : dict, list, tuple, set
        Object to hash.

    Returns
    -------
    int
        Hash of the object.
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    # Convert the hex back to int and restrict it to the relevant int range
    return int(hash_as_hex, 16) % 4294967295


def load_gudhi_dataset(
    cfg: omegaconf.DictConfig,
    feature_generator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    target_generator: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch_geometric.data.Data:
    """Load a dataset from the gudhi.datasets module."""
    if not cfg.data_name.startswith("gudhi_"):
        raise ValueError(
            "This function should only be used with gudhi datasets"
        )

    gudhi_dataset_name = cfg.data_name.removeprefix("gudhi_")

    if gudhi_dataset_name == "sphere":
        points_data = points.sphere(
            n_samples=cfg["n_samples"],
            ambient_dim=cfg["ambient_dim"],
            sample=cfg["sample"],
        )
    elif gudhi_dataset_name == "torus":
        points_data = points.torus(
            n_samples=cfg["n_samples"], dim=cfg["dim"], sample=cfg["sample"]
        )
    elif gudhi_dataset_name == "bunny":
        file_path = osp.join(
            rootutils.find_root(), cfg["data_dir"], "bunny", "bunny.npy"
        )
        points_data = fetch_bunny(
            file_path=file_path,
            accept_license=cfg.get("accept_license", False),
        )
    elif gudhi_dataset_name == "spiral_2d":
        file_path = osp.join(
            rootutils.find_root(),
            cfg["data_dir"],
            "spiral_2d",
            "spiral_2d.npy",
        )
        points_data = fetch_spiral_2d(file_path=file_path)
    elif gudhi_dataset_name == "daily_activities":
        file_path = osp.join(
            rootutils.find_root(),
            cfg["data_dir"],
            "activities",
            "activities.npy",
        )
        data = fetch_daily_activities(file_path=file_path)
        points_data = data[:, :3]
    else:
        raise ValueError(f"Gudhi dataset {gudhi_dataset_name} not recognized.")

    pos = torch.tensor(points_data, dtype=torch.float)
    if feature_generator:
        x = feature_generator(pos)
        if x.shape[0] != pos.shape[0]:
            raise ValueError(
                "feature_generator must not change first dimension of points data."
            )
    else:
        x = None

    if target_generator:
        y = target_generator(pos)
        if y.shape[0] != pos.shape[0]:
            raise ValueError(
                "target_generator must not change first dimension of points data."
            )
    elif gudhi_dataset_name == "daily_activities":
        # Target is the activity type
        # 14. for 'cross_training', 18. for 'jumping', 13. for 'stepper', or 9. for 'walking'
        y = torch.tensor(data[:, 3:], dtype=torch.float)
    else:
        y = None

    return torch_geometric.data.Data(x=x, y=y, pos=pos, complex_dim=0)


def load_random_points(
    dim: int, num_classes: int, num_samples: int, seed: int = 42
) -> torch_geometric.data.Data:
    """Create a random point cloud dataset."""
    rng = np.random.default_rng(seed)

    points = torch.tensor(rng.random((num_samples, dim)), dtype=torch.float)
    classes = torch.tensor(
        rng.integers(num_classes, size=num_samples), dtype=torch.long
    )
    features = torch.tensor(
        rng.integers(2, size=(num_samples, 1)), dtype=torch.float
    )

    return torch_geometric.data.Data(
        x=features, y=classes, pos=points, complex_dim=0
    )


def load_manual_points():
    pos = torch.tensor(
        [
            [1.0, 1.0],
            [7.0, 0.0],
            [4.0, 6.0],
            [9.0, 6.0],
            [0.0, 14.0],
            [2.0, 19.0],
            [9.0, 17.0],
        ],
        dtype=torch.float,
    )
    x = torch.ones_like(pos, dtype=torch.float)
    y = torch.randint(0, 2, (pos.shape[0],), dtype=torch.float)
    return torch_geometric.data.Data(x=x, y=y, pos=pos, complex_dim=0)


def load_pointcloud_dataset(cfg):
    r"""Loads point cloud datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Point cloud dataset.
    """
    # Define the path to the data directory
    root_folder = rootutils.find_root()
    data_dir = osp.join(root_folder, cfg["data_dir"])

    if cfg["data_name"] == "random_pointcloud":
        num_points, dim = cfg["num_points"], cfg["dim"]
        pos = torch.rand((num_points, dim))
    elif cfg["data_name"] == "stanford_bunny":
        pos = fetch_bunny(
            file_path=osp.join(data_dir, "stanford_bunny.npy"),
            accept_license=False,
        )
        num_points = len(pos)
        pos = torch.tensor(pos)

    if cfg.pos_to_x:
        return torch_geometric.data.Data(
            x=pos, pos=pos, num_nodes=num_points, num_features=pos.size(1)
        )

    return torch_geometric.data.Data(
        pos=pos, num_nodes=num_points, num_features=0
    )


def load_manual_pointcloud(pos_to_x: bool = False):
    """Create a manual pointcloud for testing purposes."""
    # Define the positions
    pos = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [10, 0, 0],
            [10, 0, 1],
            [10, 1, 0],
            [10, 1, 1],
            [20, 0, 0],
            [20, 0, 1],
            [20, 1, 0],
            [20, 1, 1],
            [30, 0, 0],
        ]
    ).float()

    if pos_to_x:
        return torch_geometric.data.Data(
            x=pos, pos=pos, num_nodes=pos.size(0), num_features=pos.size(1)
        )

    return torch_geometric.data.Data(
        pos=pos, num_nodes=pos.size(0), num_features=0
    )
