import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


class PointCloudKNNLifting(PointCloud2GraphLifting):
    r"""Lifts graphs to graph domain by considering k-nearest neighbors."""

    def __init__(self, k_value=10, loop=False, **kwargs):
        self.k = k_value
        self.loop = loop

    def calculate_vector_angle(self, v1, v2):
        if v1 is None or v2 is None:
            return None
        v1 = torch.tensor(np.copy(v1), dtype=torch.float)
        v2 = torch.tensor(np.copy(v2), dtype=torch.float)
        norm_v1 = torch.linalg.norm(v1)
        norm_v2 = torch.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cos_theta = torch.dot(v1, v2) / (norm_v1 * norm_v2)
        return torch.acos(torch.clamp(cos_theta, -1.0, 1.0)) * 180 / torch.pi


    def lift_topology(self, data: Data, k=10):
        """Lifts the topology of the graph.
        Takes the point cloud data and lifts it to a graph domain by considering k-nearest neighbors
        and sequential edges.
        Moreover, as edge attributes, the distance and angle between the nodes are considered.

        Parameters
        ----------
        data : Data
            The input data containing the point cloud.
        k : int
            The number of nearest neighbors to consider.

        Returns
        -------
        dict
            The lifted topology.
        """

        coordinates = data["pos"]
        cb_vectors = data["node_attr"]

        # Sequential edges
        seq_edge_index = []
        seq_edge_attr = []
        for i in range(len(coordinates) - 1):
            seq_edge_index.append([i, i + 1])
            dist = torch.linalg.norm(coordinates[i] - coordinates[i + 1])
            angle = self.calculate_vector_angle(cb_vectors[i], cb_vectors[i + 1])
            seq_edge_attr.append([dist, angle])

        seq_edge_index = torch.tensor(seq_edge_index, dtype=torch.long).t().contiguous()

        # KNN edges
        knn_edge_index = knn_graph(coordinates, k=k)
        knn_edge_attr = []
        existing_edges = set(tuple(edge) for edge in seq_edge_index.t().tolist())

        for i, j in knn_edge_index.t().tolist():
            if (i, j) not in existing_edges and (j, i) not in existing_edges:
                dist = torch.linalg.norm(coordinates[i] - coordinates[j])
                angle = self.calculate_vector_angle(cb_vectors[i], cb_vectors[j])
                knn_edge_attr.append([dist, angle])
                existing_edges.add((i, j))
                existing_edges.add((j, i))

        knn_edge_index = torch.tensor([list(edge) for edge in existing_edges if edge not in seq_edge_index.t().tolist()], dtype=torch.long).t().contiguous()

        # Combine KNN and sequential edges
        edge_index = torch.cat([seq_edge_index, knn_edge_index], dim=1)
        edge_attr = seq_edge_attr + knn_edge_attr
        lifted_data = Data(edge_index=edge_index, edge_attr=edge_attr)

        return {
            "num_nodes": lifted_data.edge_index.unique().shape[0],
            "edge_index": lifted_data.edge_index,
            "edge_attr": edge_attr,
        }

