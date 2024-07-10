import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class ModularityMaximizationLifting(Graph2HypergraphLifting):
    def __init__(self, num_communities=2, k_neighbors=3, **kwargs):
        super().__init__(**kwargs)
        self.num_communities = num_communities
        self.k_neighbors = k_neighbors

    def modularity_matrix(self, data):
        a = torch.zeros((data.num_nodes, data.num_nodes))
        a[data.edge_index[0], data.edge_index[1]] = 1
        k = a.sum(dim=1)
        m = data.edge_index.size(1) / 2
        b = a - torch.outer(k, k) / (2 * m)
        return b

    def kmeans(self, x, n_clusters, n_iterations=100):
        # Initialize cluster centers randomly
        centroids = x[torch.randperm(x.shape[0])[:n_clusters]]

        for _ in range(n_iterations):
            # Assign points to the nearest centroid
            distances = torch.cdist(x, centroids)
            cluster_assignments = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.stack(
                [x[cluster_assignments == k].mean(dim=0) for k in range(n_clusters)]
            )

            if torch.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return cluster_assignments

    def detect_communities(self, b):
        eigvals, eigvecs = torch.linalg.eigh(b)
        leading_eigvecs = eigvecs[
            :, torch.argsort(eigvals, descending=True)[: self.num_communities]
        ]

        # Use implemented k-means clustering on the leading eigenvectors
        community_assignments = self.kmeans(leading_eigvecs, self.num_communities)
        return community_assignments

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        b = self.modularity_matrix(data)
        community_assignments = self.detect_communities(b)

        num_nodes = data.x.shape[0]
        num_hyperedges = num_nodes
        incidence_matrix = torch.zeros(num_nodes, num_nodes)

        for i in range(num_nodes):
            # Find nodes in the same community
            same_community = (
                (community_assignments == community_assignments[i]).nonzero().view(-1)
            )

            # Calculate distances to nodes in the same community
            distances = torch.norm(
                data.x[i].unsqueeze(0) - data.x[same_community], dim=1
            )

            # Select k nearest neighbors within the community
            k = min(self.k_neighbors, len(same_community))
            _, nearest_indices = torch.topk(distances, k, largest=False)
            nearest_neighbors = same_community[nearest_indices]

            # Create a hyperedge
            incidence_matrix[i, nearest_neighbors] = 1
            incidence_matrix[i, i] = 1  # Include the node itself

        incidence_matrix = incidence_matrix.to_sparse_coo()

        return {
            "incidence_hyperedges": incidence_matrix,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
