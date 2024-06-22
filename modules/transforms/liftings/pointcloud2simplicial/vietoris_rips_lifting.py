from itertools import combinations

import torch
import torch_geometric
from toponetx.classes import Simplex, SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class VietorisRipsLifting(PointCloud2SimplicialLifting):
    """Lifts point cloud data to a Vietoris-Rips Complex. It works
    by creating a 1-simplex between any two points if their distance
    is less than or equal to epsilon. It then creates an n-simplex if
    every pair of its n+1 vertices is connected by a 1-simplex.

    """

    def __init__(self, epsilon: float, **kwargs):
        assert epsilon > 0

        self.epsilon = epsilon
        super().__init__(**kwargs)

    def _get_lifted_topology(self, simplicial_complex: SimplicialComplex) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            simplicial_complex, simplicial_complex.maxdim
        )

        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )

        return lifted_topology

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        """
        Applies Vietoris-Rips lifting strategy to point cloud.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        points = data.pos

        # Calculate pairwise distance matrix between points.
        distance_matrix = torch.cdist(points, points)

        n = len(points)
        simplices = []

        # Add 0-simplices (vertices) with their associated features.
        for i in range(n):
            simplices.append(Simplex([i], features=data.x[i]))

        # Add 1-simplices (edges) where the pairwise distance between
        # points are less than epsilon
        edges = [
            [i, j]
            for i in range(n)
            for j in range(i + 1, n)
            if distance_matrix[i, j] <= self.epsilon
        ]
        simplices.extend([Simplex(edge) for edge in edges])

        # Step 3: Construct higher-dimensional simplices
        # Iteratively finds all k-dimensional simplices (starting from k = 2) that can be formed in the graph.
        k = 2
        while True:
            higher_dim_simplices = []
            for simplex in combinations(range(n), k + 1):
                if all(
                    (
                        [simplex[i], simplex[j]] in edges
                        or [simplex[j], simplex[i]] in edges
                    )
                    for i in range(k)
                    for j in range(i + 1, k + 1)
                ):
                    higher_dim_simplices.append(Simplex(list(simplex)))

            if not higher_dim_simplices:
                break

            simplices.extend(higher_dim_simplices)
            k += 1

        SC = SimplicialComplex(simplices)

        return self._get_lifted_topology(SC)


if __name__ == "__main__":
    from modules.data.load.loaders import PointCloudLoader

    transform = VietorisRipsLifting(epsilon=0.5)

    dataloader = PointCloudLoader(
        {
            "num_classes": 3,
            "data_dir": "/Users/elphicm/PycharmProjects/challenge-icml-2024/modules/transforms/liftings/pointcloud2simplicial/",
        }
    )
    data = dataloader.load()[0]

    transform(data)
