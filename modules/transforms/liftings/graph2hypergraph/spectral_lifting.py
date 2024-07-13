import warnings

import torch
import torch_geometric
from sklearn.cluster import KMeans

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class SpectralLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by finding clusters using spectral clustering [Ng, Jordan, and Weiss (2002) [[1]](https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html)]

    Parameters
    ----------
    n_c : int, optional
        The number of clusters. Default is None.
    cluster_alg : str, optional
        The clustering algorithm to use after spectral projection. Default is KMeans.
    eps : float, optional
        The threshold to use on the eigenvalues before inverting the matrix to avoid division by 0. Default is 1e-
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        n_c=None,
        cluster_alg="KMeans",
        eps=1e-9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_c = n_c
        self.eps = eps
        self.init_clust_alg(cluster_alg)

    def init_clust_alg(self, cluster_alg: str):
        cluster_algs = {"KMeans": KMeans}
        if cluster_alg not in cluster_algs:
            warnings.warn(
                f"KMeans will be used since the algorithm {cluster_alg} was not recognized. Available algorithms: {list(cluster_algs.values())}"
            )
        self.cluster_alg = cluster_algs.get(cluster_alg, KMeans)

    def get_degree_matrix(self, W: torch.Tensor):
        return torch.diag(W.sum(1))

    def get_sqrt_inv(self, D: torch.Tensor):
        degrees = torch.clamp(torch.diag(D), min=self.eps)
        return torch.diag(degrees ** (-0.5))

    def to_dense(self, m: torch.Tensor):
        M = torch.zeros(self.num_nodes, self.num_nodes)
        n_edges = m.shape[1]
        for i in range(n_edges):
            M[m[0, i], m[1, i]] = 1
            M[m[1, i], m[0, i]] = 1
        return M

    def get_normalized_laplacian(self, W: torch.Tensor):
        # Retrieve dense representation of W
        W = self.to_dense(W)

        # Compute degree matrix
        D = self.get_degree_matrix(W)

        # Compute unnormalized Laplacian
        L = D - W

        # Compute the square root of the inverse of the degree matrix
        Dinv = self.get_sqrt_inv(D)

        # Return normalized Laplacian
        return Dinv @ L @ Dinv

    def find_gaps(self, v: torch.Tensor, a_max: int = 6, a_min: int = 2):
        def find_gap(a):
            """Finds index largest gap using the eigengap heuristic"""
            for k in range(1, len(v)):
                m = v[:k].mean()
                s = v[:k].std()
                if v[k] > m + s * a:
                    return k - 1

        def find_k_largest_diff():
            """Finds index largest gap usings absolute differences"""
            max_diff = 0
            max_index = 0
            for i in range(1, len(v)):
                diff = abs(v[i] - v[i - 1])
                if diff > max_diff:
                    max_diff = diff
                    max_index = i
            return max_index - 1

        # Attempt to find index largest gap using eigengap heuristic.
        for a in range(a_max, a_min - 1, -1):
            k = find_gap(a)
            if k and k > 1:
                break

        if k is None or k == 1:
            # Fall back to using largest absolute difference in case gap heuristic did not work
            warnings.warn(
                "Unable to confidently determine the number of clusters n_c. The largest difference between consecutive eigenvalues was used to determine the number of clusters. Please provide n_c."
            )
            k = find_k_largest_diff()
        if k == 1:
            warnings.warn(
                "Please provide the number of clusters n_c. The heuristics identified a single cluster and n_c was set to 2."
            )
            k += 1
        return k

    def get_first_eigenvectors(self, Lsym: torch.Tensor):
        # Compute eigenvectors
        Lambdas, V = torch.linalg.eig(Lsym)
        Lambdas, ind_sort = torch.sort(torch.abs(Lambdas))

        # Filter eigenvectors
        if not self.n_c:
            self.n_c = self.find_gaps(Lambdas)

        # Return eigenvectors associated to the self.nc smallest eigenvalues
        return torch.abs(V[:, ind_sort[: self.n_c + 1]])

    def normalize_rows(self, U: torch.Tensor):
        return U / ((U**2).sum(1, keepdims=True).sqrt())

    def build_clusters(self, U: torch.Tensor):
        return torch.tensor(self.cluster_alg(n_clusters=self.n_c).fit(U).labels_)

    def build_incidence_hyperedges(self, indices_clusters):
        A = torch.zeros(self.n_c, self.num_nodes)
        for c in range(self.n_c):
            ind_curr_clust = torch.nonzero(indices_clusters == c)
            A[c, ind_curr_clust] = 1
        return A

    def transform(self, data: torch_geometric.data.Data) -> dict:
        # Compute normalized Laplacian
        self.num_nodes = data.x.shape[0]
        Lsym = self.get_normalized_laplacian(data.edge_index)

        # Compute eigenvectors matrix
        U = self.get_first_eigenvectors(Lsym)

        # Normalize rows
        U = self.normalize_rows(U)

        # Build clusters
        indices_clusters = self.build_clusters(U)

        # Return incidence_hyperedges matrix
        return self.build_incidence_hyperedges(indices_clusters=indices_clusters).T

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts graphs to hypergraph domain by finding clusters using spectral clustering [Ng, Jordan, and Weiss (2002) [[1]](https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html)]

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """

        data.pos = data.x
        incidence_1 = self.transform(data).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": self.n_c,
            "x_0": data.x,
        }
