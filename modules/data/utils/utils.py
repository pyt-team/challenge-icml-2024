import hashlib
import os.path as osp
import pickle

import networkx as nx
import numpy as np
import omegaconf
import rootutils
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from gudhi.datasets.remote import fetch_bunny
from topomodelx.utils.sparse import from_sparse
from torch_geometric.data import Data
from torch_sparse import coalesce

from modules.data.utils.custom_dataset import CustomDataset


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
                    connectivity[
                        f"{connectivity_info}_{rank_idx}"
                    ] = generate_zero_sparse_connectivity(
                        m=practical_shape[rank_idx - 1], n=practical_shape[rank_idx]
                    )
                else:
                    connectivity[
                        f"{connectivity_info}_{rank_idx}"
                    ] = generate_zero_sparse_connectivity(
                        m=practical_shape[rank_idx], n=practical_shape[rank_idx]
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
    if cfg["data_name"] != "KarateClub":
        return NotImplementedError
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

    edge_idx = 0  # num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph:
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    # check that every node is in some hyperedge
    if len(np.unique(node_list)) != num_nodes:
        # add self hyperedges to isolated nodes
        isolated_nodes = np.setdiff1d(np.arange(num_nodes), np.unique(node_list))

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
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
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
    root_folder = rootutils.find_root()
    data_dir = osp.join(root_folder, cfg["data_dir"])

    if cfg["data_name"] == "random_pointcloud":
        num_points, dim = cfg["num_points"], cfg["dim"]
        pos = torch.rand((num_points, dim))
    elif cfg["data_name"] == "annulus":
        num_points, dim = cfg["num_points"], cfg["dim"]
        pos = annulus_2d(dim, num_points)
    elif cfg["data_name"] == "stanford_bunny":
        pos = fetch_bunny(
            file_path=osp.join(data_dir, "stanford_bunny.npy"),
            accept_license=False,
        )
        pos = torch.tensor(pos)

    return CustomDataset(
        [
            torch_geometric.data.Data(
                pos=pos,
            )
        ],
        data_dir,
    )


def annulus_2d(D, N, R1=0.8, R2=1, A=0):
    n = 0
    P = np.array([[0.0] * D] * N)

    rng = np.random.default_rng()
    while n < N:
        p = rng.uniform(-R2, R2, D)
        if np.linalg.norm(p) > R2 or np.linalg.norm(p) < R1:
            continue
        if (p[0] > 0) and (np.abs(p[1]) < A / 2):
            continue
        P[n] = p
        n = n + 1
    return torch.tensor(P)


def load_annulus():
    pos = annulus_2d(2, 1000)
    return torch_geometric.data.Data(pos=pos)


def load_manual_pointcloud():
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

    return torch_geometric.data.Data(pos=pos, num_nodes=pos.size(0), num_features=0)


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
