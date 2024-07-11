from itertools import combinations

import gudhi
import gudhi.simplex_tree
import networkx as nx
import numpy as np
import torch
from toponetx.classes import SimplicialComplex
from torch_geometric.data import Data

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class RandomFlagComplexLifting(PointCloud2SimplicialLifting):
    """  Lifting of pointclouds to simplicial complexes using the Random Flag Complex construction.
    """
    def __init__(self, steps, alpha=None, p=None, **kwargs):
        self.alpha = alpha
        self.steps = steps
        self.p = p
        super().__init__(**kwargs)

    def lift_topology(self, data: Data) -> dict:
        # Get the number of points and generate an empty graph
        n = data["x"].size(0)
        if self.p is None:
            self.p = np.power(n, -self.alpha)
        print(self.p)

        adj_mat = np.zeros((n, n))
        indices = np.tril_indices(n)

        st = gudhi.SimplexTree()

        # For each step, sample from random binomial distribution
        # for each edge appearign
        for _ in range(self.steps):
            number_of_edges = n*(n+1)//2
            prob = np.random.RandomState.binomial(1, self.p, size=number_of_edges)
            tmp_mat = np.zeros((n, n))
            tmp_mat[indices] = prob
            np.logical_or(adj_mat, tmp_mat, out=adj_mat)
        np.fill_diagonal(adj_mat, 0)

        # Insert all vertices
        for i in range(n):
            st.insert([i])

        graph: nx.Graph = nx.from_numpy_matrix(adj_mat).to_undirected()

        # Insert all edges
        for v, u in graph.edges:
            st.insert([v, u])

        simplicial_complex = SimplicialComplex(graph)

        # Add features to the vertices
        feats = {
            i: f
            for i, f in enumerate(data["x"])
        }

        simplicial_complex.set_simplex_attributes(feats, name="features")

        # Find the cliques up to the maximum dimension specified
        cliques = nx.find_cliques(graph)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]

        for clique in cliques:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))


        # Add the k-tuples as simplices
        for set_k_simplices in simplices:
            for k_simplex in set_k_simplices:
                st.insert(k_simplex)
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, st)

    def _get_lifted_topology(self, simplicial_complex: SimplicialComplex, st: gudhi.SimplexTree) -> dict:

        # Connectivity of the complex
        lifted_topology = get_complex_connectivity(
            simplicial_complex, self.complex_dim, signed=False
        )
        # Computing the persitence to obtain the Betti numbers
        st.compute_persistence(persistence_dim_max=True)

        # Save the Betti numbers in the Data object
        lifted_topology["betti"] = torch.tensor(st.betti_numbers())


        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )

        # Add the indices of the simplices
        for r in range(simplicial_complex.dim):
            lifted_topology[f"x_idx_{r}"] = torch.tensor(simplicial_complex.skeleton(r))

        return lifted_topology
