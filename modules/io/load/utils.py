import hashlib
import os
import os.path as osp
import pickle

import numpy as np
import omegaconf
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from torch_geometric.data import Data
from torch_sparse import coalesce

# from sklearn.model_selection import StratifiedKFold
# from modules.data.datasets import CustomDataset


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
            except ValueError:
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
    pass


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
            features["x_{}".format(rank_idx)] = torch.tensor(
                np.stack(
                    list(
                        data.get_simplex_attributes(
                            dict_feat_equivalence[rank_idx]
                        ).values()
                    )
                )
            )
        except:
            features["x_{}".format(rank_idx)] = torch.tensor(
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

    # TODO: Fix the splits
    data = torch_geometric.transforms.random_node_split.RandomNodeSplit(
        num_val=4, num_test=4
    )(data)
    return data


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
    for he in hypergraph.keys():
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
    data_lst = [data for data in dataset]
    return data_lst


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
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, (str, int, float, bool, type(None))):
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
    # convert the hex back to int and restrict it to the relevant int range
    seed = int(hash_as_hex, 16) % 4294967295
    return seed

def plot_manual_graph(data):
    import numpy as np
    import torch
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    
    # Collect vertices
    vertices = [i for i in range(data.x.shape[0])]

    # Collect edges
    edges = []
    edge_mapper = {}
    for edge_idx, edge in enumerate(abs(data.incidence_1.to_dense().T)):
        node_idxs = torch.where(edge != 0)[0].numpy()
        
        edges.append(torch.where(edge != 0)[0].numpy())
        edge_mapper[edge_idx] = sorted(node_idxs)


    # Collect triangles
    triangles = []
    triangle_mapper = {}
    for triangle_idx, triangle in enumerate(abs(data.incidence_2.to_dense().T)):
        edge_idxs = torch.where(triangle != 0)[0].numpy()
        
        nodes = []
        for edge_idx in edge_idxs:
            nodes += (edge_mapper[edge_idx])
        
        triangle_mapper[triangle_idx] = {
            'edge_idxs': sorted(edge_idxs),
            'node_idxs': sorted(list(set(nodes)))
                                        
        }
        
        triangles.append(sorted(list(set(nodes))))

    # Collect tetraherdons
    tetrahedrons = []
    tetrahedron_mapper = {}
    for tetrahedron_idx, tetrahedron in enumerate(abs(data.incidence_3.to_dense().T)):
        triangle_idxs = torch.where(tetrahedron != 0)[0].numpy()
        
        nodes = []
        edges_in_tetrahedrons = []
        for triangle_idx in triangle_idxs:
            nodes += (triangle_mapper[triangle_idx]['node_idxs'])
            edges_in_tetrahedrons += (triangle_mapper[triangle_idx]['edge_idxs'])
        
        tetrahedron_mapper[tetrahedron_idx] = {
            'triangle_idxs': sorted(triangle_idxs),
            'edge_idxs': sorted(list(set(edges_in_tetrahedrons))),
            'node_idxs': sorted(list(set(nodes)))
                                        
        }   
        
        tetrahedrons.append(sorted(list(set(nodes))))

    edges = np.array(edges)
    triangles = np.array(triangles)
    tetrahedrons = np.array(tetrahedrons)

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)

    # Plot the graph with edge indices using other layout
    pos = nx.spring_layout(G, seed=42)
    #pos[3] = np.array([0.15539556, 0.25])

    plt.figure(figsize=(5, 5))
    # Draw the graph with labels
    nx.draw(
        G,
        pos,
        labels={i: f"v_{i}" for i in G.nodes()},
        node_size=500,
        node_color="skyblue",
        font_size=12,
        edge_color='black', 
        width=1, 
        linewidths=1,
        alpha=0.9,
        
    )

    # Color the faces (triangles) of the graph
    face_color_map = {
        0: "pink",
        1: "gray",
        2: "blue",
        3: "blue",
        4: "orange",
        5: "purple",
        6: "red",
        7: "brown",
        8: "black",
        9: "gray",
    }

    for i, clique in enumerate(triangles):
        # Get the face color:
        # Calculate to how many tetrahedrons cique belongs
        # Then assign the color to the face
        counter = 0
        for tetrahedron in tetrahedrons:
            from itertools import combinations

            for comb in combinations(tetrahedron, 3):
                if set(clique) == set(comb):
                    counter += 1

        polygon = [pos[v] for v in clique]
        poly = Polygon(
            polygon,
            closed=True,
            facecolor=face_color_map[counter],
            #edgecolor="pink",
            alpha=0.3,
        )
        plt.gca().add_patch(poly)



    # Draw edges with different color and thickness
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={tuple(corr_nodes): f'e_{edge_idx}' for edge_idx, corr_nodes in edge_mapper.items()},
        font_color='red',
        alpha=0.9,
        font_size=8,
        rotate=False,
        horizontalalignment='center',
        verticalalignment='center'
    )


    plt.title("Graph with cliques colored (8 vertices)")
    plt.axis('off')
    plt.show()
