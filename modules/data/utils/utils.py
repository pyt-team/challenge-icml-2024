import hashlib
import os.path as osp
import pickle

from typing import Dict
import networkx as nx
import numpy as np
import omegaconf
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from torch_geometric.data import Data
from torch_sparse import coalesce

class SimplexData(torch_geometric.data.Data):
    def __inc__(self, key: str, value, *args, **kwargs):
        if 'adjacency' in key:
            rank = int(key.split('_')[-1])
            return torch.tensor([getattr(self, f'x_{rank}').size(0)])
            #return torch.tensor([[getattr(self, f'x_{rank}').size(0)], [getattr(self, f'x_{rank}').size(0)]])
        elif 'incidence' in key:
            rank = int(key.split('_')[-1])
            if rank == 0:
                return torch.tensor([getattr(self, f'x_{rank}').size(0)])
            return torch.tensor([[getattr(self, f'x_{rank-1}').size(0)], [getattr(self, f'x_{rank}').size(0)]])
        elif key == 'x_0' or key == 'x_idx_0':
            return torch.tensor([getattr(self, f'x_0').size(0)])
        elif key == 'x_1' or key == 'x_idx_1':
            return torch.tensor([getattr(self, f'x_0').size(0)])
        elif key == 'x_2' or key == 'x_idx_2':
            return torch.tensor([getattr(self, f'x_0').size(0)])
        elif 'index' in key:
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if 'adjacency' in key or 'incidence' in key or 'index' in key:
            return 1
        else:
            return 0


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

def find_non_overlap(tensor1, tensor2):
    """
    
    """

    # Initialize an output boolean tensor with the same shape as tensor2
    output_tensor = torch.zeros_like(tensor2, dtype=torch.bool)
    
    # Loop through each row
    for i, (row1, row2) in enumerate(zip(tensor1, tensor2)):
        # Convert the rows to sets
        set1 = set(row1.tolist())
        set2 = set(row2.tolist())
        
        # Find the symmetric difference (which should be a single element)
        non_overlapping_set = set1.symmetric_difference(set2)

        # There should be exactly one non-overlapping item
        if len(non_overlapping_set) == 1:
            non_overlapping_item = list(non_overlapping_set)[0]
            # Find the position of the non-overlapping item in tensor2's row
            pos = row2.tolist().index(non_overlapping_item)
            # Set the corresponding position in the output tensor to True
            output_tensor[i, pos] = True
        else:
            print(set1)
            print(set2)
            raise ValueError("There should be exactly one non-overlapping item per row.")
# Convert the result list to a tensor
    return output_tensor

def compute_invariance_r_to_r(simplices: Dict[int, torch.Tensor], pos: torch.Tensor, adj: Dict[int, torch.Tensor]):
    """ 
        Computes the invariances from r-cells to r-cells geometrical properties
        Parameters
        ----------
        simplices : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, r)
            Indices of each component of a simplex with respect to the nodes in the original graph 
        pos : torch.Tensor, shape = (n_rank_0_cells, 3)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        adj : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, n_rank_r_cells)
            Adjacency matrices :math:`H_r` mapping cells to cells via lower and upper cells.

    """

    adj_dict={}
    sending_nodes = adj[0][0]
    receiving_nodes = adj[0][1]
    distance = torch.linalg.norm(pos[sending_nodes] - pos[receiving_nodes], dim=1)
    # Ugly fix for initializing to correct device
    area = torch.zeros(len(distance), 2).to(distance.device)
    adj_dict[0] = torch.cat((distance.unsqueeze(1), area),dim=1)
    
    #for 1 dimensional simplexes
    
    
    sending_simplices = adj[1][0]
    receiving_simplices = adj[1][1]
    
    connected_nodes = simplices[1][receiving_simplices].squeeze(1)
    sending_simplex_nodes = simplices[1][sending_simplices]
    
    vecs = pos[simplices[1][:, 0]] - pos[simplices[1][:, 1]]
    send_vec, rec_vec = vecs[sending_simplices], vecs[receiving_simplices]
    send_norm, rec_norm = torch.linalg.norm(send_vec, ord=2, dim=1), torch.linalg.norm(rec_vec, ord=2, dim=1)
    dot = torch.sum(send_vec * rec_vec)
    cos_angle = dot / (send_norm * rec_norm)
    eps = 1e-6
    #numerical stability
    angle = torch.arccos(cos_angle.clamp(-1 + eps, 1 - eps))
    adj_dict[1] = torch.stack([
            #pi - a
            torch.linalg.norm(pos[sending_simplex_nodes[:,0]] - pos[sending_simplex_nodes[:,1]], dim=1),
            #pi - b
            torch.linalg.norm(pos[sending_simplex_nodes[:,0]] - pos[connected_nodes[:,1]], dim=1),
            #a - b
            torch.linalg.norm(pos[sending_simplex_nodes[:,1]] - pos[connected_nodes[:,1]], dim=1),
            #area sending (same as distance for edges)
            torch.linalg.norm(pos[sending_simplex_nodes[:,0]] - pos[sending_simplex_nodes[:,1]], dim=1),
            #are receiving (same as distance for edges)
            torch.linalg.norm(pos[sending_simplex_nodes[:, 0]] - pos[connected_nodes[:, 1]], dim=1),
            angle
        ], dim=1)    
    
    return adj_dict
def compute_invariance_r_minus_1_to_r(simplices: Dict[int, torch.Tensor], pos: torch.Tensor, inc: Dict[int, torch.Tensor]):
    """ 
        Computes the invariances from r-cells to r-cells geometrical properties
        Parameters
        ----------
        simplices : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, r)
            Indices of each component of a simplex with respect to the nodes in the original graph 
        pos : torch.Tensor, shape = (n_rank_0_cells, 3)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        inc : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_minus_1_cells, n_rank_r_cells)
            Adjacency matrices :math:`I^1_r` with weights of map from r-cells to (r-1)-cells

    """
    inc_dict={}
    # for 0_1 dimensional simplices:
    result_matrix_0_1 = []
    sending_nodes = inc[0][0]
    receiving_simplices = inc[0][1]
    sending_edges = inc[1][0]
    nodes_of_receiving_simplices = simplices[1][receiving_simplices]
    mask = nodes_of_receiving_simplices != sending_nodes.unsqueeze(1)
    receiving_nodes = nodes_of_receiving_simplices[mask]
    
    distance = torch.linalg.norm(pos[sending_nodes] - pos[receiving_nodes], dim=1)
    features = distance.view(-1, 1).repeat(1, 3)
    #area of the sending node is 0.
    features[:,1] = 0
    inc_dict[0] = features
    
    receiving_triangles = inc[1][1]
    
    triangle_nodes = simplices[2][receiving_triangles]
    sending_edge_nodes = simplices[1][sending_edges]   
    a_index = find_non_overlap(sending_edge_nodes,triangle_nodes)
    
    p1, p2, a= pos[sending_edge_nodes[:,0]], pos[sending_edge_nodes[:,1]], pos[triangle_nodes[a_index]]
    v1, v2, b = p1 - a, p2 - a, p1 - p2
    eps = 1e-6
    v1_n, v2_n, b_n = torch.linalg.norm(v1, dim=1), torch.linalg.norm(v2, dim=1), torch.linalg.norm(b, dim=1)
    v1_a = torch.arccos((torch.sum(v1 * b, dim=1) / (v1_n * b_n)).clamp(-1 + eps, 1 - eps))
    v2_a = torch.arccos((torch.sum(v2 * b, dim=1) / (v2_n * b_n)).clamp(-1 + eps, 1 - eps))
    b_a = torch.arccos((torch.sum(v1 * v2, dim=1) / (v1_n * v2_n)).clamp(-1 + eps, 1 - eps))
    #check if dimensions of angle are appropriate [n,2]
    angle = torch.moveaxis(torch.vstack((v1_a + v2_a, b_a)), 0, 1)
    
    area_1 = b_n.unsqueeze(1)
    
    area_2 = (torch.norm(torch.cross(pos[simplices[2][receiving_triangles,0]] - pos[simplices[2][receiving_triangles, 1]],
                                        pos[simplices[2][receiving_triangles, 0]] - pos[simplices[2][receiving_triangles, 2]], dim=1),
                            dim=1) / 2).unsqueeze(1)
    distances = torch.stack([
            torch.linalg.norm(p1 - a, dim=1)
            + torch.linalg.norm(p1 - a, dim=1),
            torch.linalg.norm(p2 - a, dim=1)
        ], dim=1)
    inc_dict[1] = torch.cat((distances,area_1,area_2,angle),dim=1)
    return inc_dict 