from itertools import combinations

import torch
import torch_geometric
from toponetx.classes import Simplex, SimplicialComplex

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

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        points = data.pos

        distance_matrix = torch.cdist(points, points)

        n = len(points)
        simplices = []

        features = data.keys()
        # Add 0-simplices (vertices)
        for i in range(n):
            # ['a':data[feature] for feature in features]
            simplices.append(Simplex([i], features=data.x[i]))

        # Add 1-simplices (edges)
        edges = [
            [i, j]
            for i in range(n)
            for j in range(i + 1, n)
            if distance_matrix[i, j] <= self.epsilon
        ]
        simplices.extend([Simplex(edge) for edge in edges])

        # Step 3: Construct higher-dimensional simplices
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

        simplicial_complex = SimplicialComplex(simplices)

        return self._get_lifted_topology(simplicial_complex)


def plot_vietoris_rips_complex(simplicial_complex, points):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    points = points.numpy()

    # Plot 0-simplices (vertices)
    ax.scatter(points[:, 0], points[:, 1], c="blue", s=50)

    # Plot 1-simplices (edges)
    for simplex in simplicial_complex.simplices:
        if len(simplex) == 2:
            i, j = list(simplex)
            ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], "k-")

    # Plot 2-simplices (filled triangles)
    for simplex in simplicial_complex.simplices:
        if len(simplex) == 3:
            i, j, k = list(simplex)
            triangle = plt.Polygon(
                [points[i], points[j], points[k]], color="gray", alpha=0.5
            )
            ax.add_patch(triangle)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Vietoris-Rips Complex")
    plt.show()


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
