import pprint
import random
import shutil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import omegaconf
import rootutils
import torch
import torch_geometric
from matplotlib.patches import Polygon

plt.rcParams["text.usetex"] = bool(shutil.which("latex"))
rootutils.setup_root("./", indicator=".project-root", pythonpath=True)


def load_dataset_config(dataset_name: str) -> omegaconf.DictConfig:
    r"""Load the dataset configuration.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    omegaconf.DictConfig
        Dataset configuration.
    """

    root_folder = rootutils.find_root()
    dataset_config_path = f"{root_folder}/configs/datasets/{dataset_name}.yaml"
    dataset_config = omegaconf.OmegaConf.load(dataset_config_path)

    # Print configuration
    print(f"\nDataset configuration for {dataset_name}:\n")
    pprint.pp(dict(dataset_config.copy()))
    return dataset_config


def load_transform_config(
    transform_type: str, transform_id: str
) -> omegaconf.DictConfig:
    r"""Load the transform configuration.

    Parameters
    ----------
    transform_name : str
        Name of the transform.
    transform_id : str
        Identifier of the transform. If the transform is a topological lifting,
        it should include the type of the lifting and the identifier separated by a '/'
        (e.g. graph2cell/cycle_lifting).

    Returns
    -------
    omegaconf.DictConfig
        Transform configuration.
    """
    root_folder = rootutils.find_root()
    transform_config_path = (
        f"{root_folder}/configs/transforms/{transform_type}/{transform_id}.yaml"
    )
    transform_config = omegaconf.OmegaConf.load(transform_config_path)
    # Print configuration
    if transform_type == "liftings":
        print(f"\nTransform configuration for {transform_id}:\n")
    else:
        print(f"\nTransform configuration for {transform_type}/{transform_id}:\n")
    pprint.pp(dict(transform_config.copy()))
    return transform_config


def load_model_config(model_type: str, model_name: str) -> omegaconf.DictConfig:
    r"""Load the model configuration.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    omegaconf.DictConfig
        Model configuration.
    """
    root_folder = rootutils.find_root()
    model_config_path = f"{root_folder}/configs/models/{model_type}/{model_name}.yaml"
    model_config = omegaconf.OmegaConf.load(model_config_path)
    # Print configuration
    print(f"\nModel configuration for {model_type} {model_name.upper()}:\n")
    pprint.pp(dict(model_config.copy()))
    return model_config


def describe_data(dataset: torch_geometric.data.Dataset, idx_sample: int = 0):
    r"""Describe a data sample of the considered dataset.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Data object.
    idx_sample : int
        Index of the sample to describe.
    """
    # assert isinstance(
    #     dataset, torch_geometric.data.Dataset
    # ), "Data object must be a PyG Dataset object."
    num_samples = len(dataset)
    if num_samples == 1:
        print(f"\nDataset only contains {num_samples} sample:")
    else:
        print(f"\nDataset contains {num_samples} samples.\n")
        print(
            f"Providing more details about sample {idx_sample % num_samples}/{num_samples}:"
        )
    data = dataset.get(idx_sample % num_samples)
    complex_dim = []
    features_dim = []
    # If lifted, we can look at the generated features for each cell
    for dim in range(10):
        if hasattr(data, f"x_{dim}") and getattr(data, f"x_{dim}").shape[0] > 0:
            complex_dim.append(getattr(data, f"x_{dim}").shape[0])
            features_dim.append(getattr(data, f"x_{dim}").shape[1])
    # If not lifted, we check the classical fields of a dataset loaded from PyG
    if len(complex_dim) == 0:
        if hasattr(data, "num_nodes"):
            complex_dim.append(data.num_nodes)
            features_dim.append(data.num_node_features)
        elif hasattr(data, "x"):
            complex_dim.append(data.x.shape[0])
            features_dim.append(data.x.shape[1])
        else:
            raise ValueError("Data object does not contain any vertices/points.")
        if hasattr(data, "num_edges"):
            complex_dim.append(data.num_edges)
            features_dim.append(data.num_edge_features)
        elif hasattr(data, "edge_index") and data.edge_index is not None:
            complex_dim.append(data.edge_index.shape[1])
            features_dim.append(data.edge_attr.shape[1])
    # Check if the data object contains hyperedges
    hyperedges = False
    if hasattr(data, "x_hyperedges"):
        hyperedges = data.x_hyperedges.shape[0]
        hyperedges_features_dim = data.x_hyperedges.shape[1]

    # Plot the graph if it is not too large
    if complex_dim[0] < 50:
        plot_manual_graph(data)

    if hyperedges:
        print(
            f" - Hypergraph with {complex_dim[0]} vertices and {hyperedges} hyperedges."
        )
        features_dim.append(hyperedges_features_dim)
        print(f" - The nodes have feature dimensions {features_dim[0]}.")
        print(f" - The hyperedges have feature dimensions {features_dim[1]}.")
    else:
        if len(complex_dim) == 1:
            print(f" - Set with {complex_dim[0]} points.")
            print(f" - Features dimension: {features_dim[0]}")
        elif len(complex_dim) == 2:
            print(
                f" - Graph with {complex_dim[0]} vertices and {complex_dim[1]} edges."
            )
            print(f" - Features dimensions: {features_dim}")
            # Check if there are isolated nodes
            if hasattr(data, "edge_index") and hasattr(data, "x"):
                connected_nodes = torch.unique(data.edge_index)
                isolated_nodes = []
                for i in range(data.x.shape[0]):
                    if i not in connected_nodes:
                        isolated_nodes.append(i)  # noqa : PERF401
                print(f" - There are {len(isolated_nodes)} isolated nodes.")
        else:
            for i, c_d in enumerate(complex_dim):
                print(f" - The complex has {c_d} {i}-cells.")
                print(f" - The {i}-cells have features dimension {features_dim[i]}")
    print("")


def plot_manual_graph(data, title=None):
    r"""Plot a manual graph. If lifted, the plot shows the inferred
    higher-order structures (bipartite graph for hyperedges,
    colored 2-cells and 3-cells for simplicial/cell complexes).
    Combinatorial complexes are plotted as simplicial/cell complexes.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Data object containing the graph.
    title: str
        Title for the plot.
    """

    def sort_vertices_ccw(vertices):
        r"""Sort vertices in counter-clockwise order.

        Parameters
        ----------
        vertices : list
            List of vertices.

        Returns
        -------
        list
            List of vertices sorted in counter-clockwise order.
        """
        centroid = [
            sum(v[0] for v in vertices) / len(vertices),
            sum(v[1] for v in vertices) / len(vertices),
        ]
        return sorted(
            vertices, key=lambda v: (np.arctan2(v[1] - centroid[1], v[0] - centroid[0]))
        )

    max_order = 1
    if hasattr(data, "incidence_3"):
        max_order = 3
    elif hasattr(data, "incidence_2"):
        max_order = 2
    elif hasattr(data, "incidence_hyperedges"):
        max_order = 0
        incidence = data.incidence_hyperedges.coalesce()

    # Collect vertices
    vertices = [i for i in range(data.x.shape[0])]

    # Hyperedges
    if max_order == 0:
        n_vertices = len(vertices)
        n_hyperedges = incidence.shape[1]
        vertices += [i + n_vertices for i in range(n_hyperedges)]
        indices = incidence.indices()
        edges = np.array([indices[1].numpy(), indices[0].numpy() + n_vertices]).T
        pos_n = [[i, 0] for i in range(n_vertices)]
        pos_he = [[i, 1] for i in range(n_hyperedges)]
        pos = pos_n + pos_he

    # Collect edges
    if max_order > 0:
        edges = []
        edge_mapper = {}
        if hasattr(data, "incidence_1"):
            for edge_idx, edge in enumerate(abs(data.incidence_1.to_dense().T)):
                node_idxs = torch.where(edge != 0)[0].numpy()

                edges.append(torch.where(edge != 0)[0].numpy())
                edge_mapper[edge_idx] = sorted(node_idxs)
            edges = np.array(edges)
        elif hasattr(data, "edge_index") and data.edge_index is not None:
            edges = data.edge_index.T.tolist()
            edge_mapper = {}
            for e, edge in enumerate(edges):
                edge_mapper[e] = [node for node in edge]
    # Collect 2dn order polygons
    if max_order > 1:
        faces = []
        faces_mapper = {}
        for faces_idx, face in enumerate(abs(data.incidence_2.to_dense().T)):
            edge_idxs = torch.where(face != 0)[0].numpy()

            nodes = []
            for edge_idx in edge_idxs:
                nodes += edge_mapper[edge_idx]

            faces_mapper[faces_idx] = {
                "edge_idxs": sorted(edge_idxs),
                "node_idxs": sorted(list(set(nodes))),
            }

            faces.append(sorted(list(set(nodes))))

    # Collect volumes
    if max_order == 3:
        volumes = []
        volume_mapper = {}
        for volume_idx, volume in enumerate(abs(data.incidence_3.to_dense().T)):
            face_idxs = torch.where(volume != 0)[0].numpy()

            nodes = []
            edges_in_volumes = []
            for face_idx in face_idxs:
                nodes += faces_mapper[face_idx]["node_idxs"]
                edges_in_volumes += faces_mapper[face_idx]["edge_idxs"]

            volume_mapper[volume_idx] = {
                "face_idxs": sorted(face_idxs),
                "edge_idxs": sorted(list(set(edges_in_volumes))),
                "node_idxs": sorted(list(set(nodes))),
            }

            volumes.append(sorted(list(set(nodes))))
        volumes = np.array(volumes)

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)
    # Add edges
    G.add_edges_from(edges)

    # Plot the graph with edge indices using other layout
    if max_order != 0:
        pos = nx.spring_layout(G, seed=42)
    # pos[3] = np.array([0.15539556, 0.25])

    plt.figure(figsize=(5, 5))
    # Draw the graph with labels
    if max_order == 0:
        labels = {i: "$v_{" + str(i) + "}$" for i in range(n_vertices)}
        for e in range(n_hyperedges):
            labels[e + n_vertices] = "$h_{" + str(e) + "}$"
    else:
        labels = {i: "$v_{" + str(i) + "}$" for i in G.nodes()}

    nx.draw(
        G,
        pos,
        labels=labels,
        node_size=500,
        node_color="skyblue",
        font_size=12,
        edge_color="black",
        width=1,
        linewidths=1,
        alpha=0.9,
    )

    # Color the faces of the graph
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

    if max_order > 1:
        for _, clique in enumerate(faces):
            # Get the face color:
            # Calculate to how many volumes cique belongs
            # Then assign the color to the face
            counter = 0
            if max_order == 3:
                for volume in volumes:
                    from itertools import combinations

                    for comb in combinations(volume, 3):
                        if set(clique) == set(comb):
                            counter += 1
            else:
                counter = random.randint(0, 9)

            polygon = [pos[v] for v in clique]
            sorted_polygon = sort_vertices_ccw(polygon)
            poly = Polygon(
                sorted_polygon,
                closed=True,
                fill=True,
                facecolor=face_color_map[counter],
                # edgecolor="pink",
                alpha=0.3,
            )
            plt.gca().add_patch(poly)

    # Draw edges with different color and thickness
    if max_order > 0:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={
                tuple(corr_nodes): "$e_{" + str(edge_idx) + "}$"
                for edge_idx, corr_nodes in edge_mapper.items()
            },
            font_color="red",
            alpha=0.9,
            font_size=8,
            rotate=False,
            horizontalalignment="center",
            verticalalignment="center",
        )
    if title is not None:
        plt.title(title)
    if max_order == 0:
        plt.title("Lifted Graph - top nodes represent the hyperedges")
    elif max_order == 1:
        plt.title("Original Graph")
    elif max_order == 2:
        plt.title("Lifted Graph with colored 2-cells")
    else:
        plt.title("Lifted Graph with colored 2 and 3-cells")
    plt.axis("off")
    plt.show()


def describe_simplicial_complex(data: torch_geometric.data.Data):
    r"""Describe a simplicial complex.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Data object containing the simplicial complex.
    """
    edges2nodes = [[] for _ in range(data.incidence_1.shape[1])]
    indices = data.incidence_1.coalesce().indices()
    for i in range(data.incidence_1.shape[1]):
        edges2nodes[i] = indices[0, indices[1, :] == i]
    edges2nodes = torch.stack(edges2nodes)

    triangles2edges = [[] for _ in range(data.incidence_2.shape[1])]
    indices = data.incidence_2.coalesce().indices()
    for i in range(data.incidence_2.shape[1]):
        triangles2edges[i] = indices[0, indices[1, :] == i]
    triangles2edges = torch.stack(triangles2edges)

    incidence_2 = data.incidence_2.coalesce()
    indices = incidence_2.indices()

    print(f"The simplicial complex has {incidence_2.shape[1]} triangles.")
    for triangles_idx in torch.unique(indices[1]):
        corresponding_idxs = indices[1] == triangles_idx
        edges = indices[0, corresponding_idxs]
        nodes = torch.unique(edges2nodes[edges]).numpy()
        print(f"Triangle {triangles_idx} is composed from nodes {nodes}")
        if triangles_idx >= 10:
            print("...")
            break

    incidence_3 = data.incidence_3.coalesce()
    indices = incidence_3.indices()

    print(f"\nThe simplicial complex has {incidence_3.shape[1]} tetrahedrons.")
    for tetrahedrons_idx in torch.unique(indices[1]):
        corresponding_idxs = indices[1] == tetrahedrons_idx
        triangles = indices[0, corresponding_idxs]
        nodes = torch.unique(edges2nodes[triangles2edges[triangles]]).numpy()
        print(f"Tetrahedron {tetrahedrons_idx} is composed from nodes {nodes}")
        if tetrahedrons_idx > 10:
            print("...")
            break


def describe_cell_complex(data: torch_geometric.data.Data):
    r"""Describe a simplicial complex.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Data object containing the simplicial complex.
    """
    edges2nodes = [[] for _ in range(data.incidence_1.shape[1])]
    indices = data.incidence_1.coalesce().indices()
    for i in range(data.incidence_1.shape[1]):
        edges2nodes[i] = indices[0, indices[1, :] == i]
    edges2nodes = torch.stack(edges2nodes)

    incidence_2 = data.incidence_2.coalesce()
    indices = incidence_2.indices()

    print(f"The cell complex has {incidence_2.shape[1]} cells.")
    for cell_idx in torch.unique(indices[1]):
        corresponding_idxs = indices[1] == cell_idx
        edges = indices[0, corresponding_idxs]
        nodes = torch.unique(edges2nodes[edges])
        print(f"Cell {cell_idx} is composed from the edges {nodes}")
        if cell_idx >= 10:
            print("...")
            break


def describe_hypergraph(data: torch_geometric.data.Data):
    r"""Describe a hypergraph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Data object containing the hypergraph.
    """
    incidence = data.incidence_hyperedges.coalesce()
    indices = incidence.indices()

    print(f"The hypergraph has {data.incidence_hyperedges.shape[1]} hyperedges.")
    for he_idx in torch.unique(indices[1]):
        corresponding_idxs = indices[0] == he_idx
        nodes = indices[1, corresponding_idxs]
        print(f"Hyperedge {he_idx} contains the nodes {nodes.numpy()}")
        if he_idx >= 10:
            print("...")
            break
