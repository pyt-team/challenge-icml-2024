import pprint
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import omegaconf
import rootutils
import torch
from matplotlib.patches import Polygon


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
    pprint.pp(dict(model_config.copy()))
    return model_config


def plot_manual_graph(data, title=None):
    r"""Plot a manual graph.

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
        elif hasattr(data, "edge_index"):
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
        labels = {i: f"v_{i}" for i in range(n_vertices)}
        for e in range(n_hyperedges):
            labels[e + n_vertices] = f"he_{e}"
    else:
        labels = {i: f"v_{i}" for i in G.nodes()}

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
        for i, clique in enumerate(faces):
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
                tuple(corr_nodes): f"e_{edge_idx}"
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
        plt.title("Bipartite graph. Top nodes represent the hyperedges.")
    elif max_order == 1:
        plt.title("The original graph.")
    else:
        plt.title("Graph with colored faces.")
    plt.axis("off")
    plt.show()
