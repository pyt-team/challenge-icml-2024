import hashlib
import os.path as osp
import pickle

import networkx as nx
import numpy as np
import omegaconf
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from torch_geometric.data import Data
from torch_sparse import coalesce


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
                            m=practical_shape[rank_idx - 1], n=practical_shape[rank_idx]
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
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

def load_manual_prot():
    """Create a manual graph for testing protein data.
    The graph corresponds to the representation of the
    protein with uniprotid: P0DJJ1
    """

    # Define the vertices
    vertices = [i for i in range(16)]
    y = [2005]

    # Define the edges
    edges = [
            [0, 1],
            [0, 2],
            [1, 2],
            [2, 3],
            [2, 4],
            [3, 4],
            [3, 6],
            [4, 5],
            [4, 7],
            [5, 6],
            [5, 8],
            [6, 7],
            [6, 9],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [11, 13],
            [12, 13],
            [12, 14],
            [13, 14],
            [13, 15],
            [14, 15]
        ]

    node_attr = [
        [ 1.3890,  0.6190, -0.1820],
        [-0.9270, -0.9870,  0.7330],
        [-0.2270,  1.4300, -0.5240],
        [ 1.2680,  0.3270,  0.8000],
        [ 0.1190,  1.1460, -1.0170],
        [ 0.4530, -0.8660, -1.1890],
        [ 1.4150, -0.3400,  0.5030],
        [ 0.2660,  1.5060, -0.0980],
        [-0.2630,  0.4330, -1.4480],
        [ 1.0150, -0.7640, -0.8760],
        [ 1.0040,  0.6400,  0.9630],
        [-0.7060,  1.3490, -0.2130],
        [ 0.8990, -1.1560, -0.4560],
        [-1.0300,  1.1430,  0.0910],
        [ 0.9240, -1.0550,  0.6270],
        [-1.0380,  0.6640, -0.9280]
    ]

    pos = [
        [  7.5210,   0.0560,  -6.7320],
        [  4.8200,   1.0530,  -4.2620],
        [  6.1700,   4.2550,  -2.6770],
        [  6.0640,   4.2840,   1.1930],
        [  3.1770,   6.8050,   0.7350],
        [  0.8630,   3.9710,  -0.5950],
        [  1.4180,   1.9200,   2.6160],
        [ -0.2780,   4.7140,   4.6930],
        [ -3.6400,   3.8990,   3.0330],
        [ -3.5740,   0.0960,   3.6640],
        [ -3.8460,  -0.2580,   7.5040],
        [ -7.6510,   0.2670,   7.8800],
        [ -8.4770,  -3.4030,   7.2390],
        [-11.1830,  -3.1590,   9.8940],
        [-12.1290,  -6.7670,  10.4510],
        [-15.7920,  -5.9970,  11.1970]
    ]

    edge_attr = [
        [3.7934558, 149.50481451169998],
        [5.9916463, 73.61527190033692],
        [3.8193624, 131.95556949748686],
        [3.8715599, 95.81568896389793],
        [5.2059865, 24.99968085162557],
        [3.8600485, 97.01387095949963],
        [5.403586, 28.039804503840163],
        [3.892949, 83.4287695658028],
        [5.6546497, 37.945642805792026],
        [3.850344, 81.81569949178396],
        [5.7831287, 58.674074660579485],
        [3.872568, 94.49543679831403],
        [5.4171343, 58.10693508314127],
        [3.8370392, 72.06193427289432],
        [3.8555577, 73.54177428094899],
        [3.8658636, 97.62340401695313],
        [3.8594074, 91.23113911037706],
        [3.8160264, 152.7860125877105],
        [5.316831, 18.30498413525865],
        [3.7988148, 165.50490996053213],
        [5.9135895, 41.512625041705014],
        [3.771317, 152.5153703231938],
        [5.56731, 42.83026706257669],
        [3.816672, 161.06592876960846]
        ]

    # Create a graph
    G = nx.Graph()
    # Add vertices
    G.add_nodes_from(vertices)
    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    x = torch.tensor([[1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.],
        [1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0.,
         0., 0.]])

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
        edge_attr=torch.tensor(edge_attr),
        node_attr=torch.tensor(node_attr),
        pos=torch.tensor(pos)
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
