from functools import partial

import gudhi
import gudhi.cover_complex
import numpy as np
import statsmodels.stats.multitest as mt
import torch
import torch_geometric
from statsmodels.distributions.empirical_distribution import ECDF
from torch_geometric.utils.convert import from_networkx

from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting

rng = np.random.default_rng()


def persistent_homology(points: torch.Tensor, subcomplex_inds: list[int] | None = None):
    """Calculate (relative) persistent homology using Alpha complex.

    Parameters
    ----------
    points : torch.Tensor
        Set of points.
    subcomplex_inds : list[int] | None, optional
        Points on the boundary (subcomplex), by default None

    Returns
    -------
    torch.Tensor
        Persistence diagram
    """
    st = gudhi.AlphaComplex(points=points).create_simplex_tree()

    if subcomplex_inds is not None:
        subcomplex = [
            simplex
            for simplex, _ in st.get_simplices()
            if all(x in subcomplex_inds for x in simplex)
        ]

        new_vertex = st.num_vertices()
        st.insert([new_vertex], 0)
        for simplex in subcomplex:
            st.insert([*simplex, new_vertex], st.filtration(simplex))

    persistence = st.persistence()

    return np.array(
        [(birth, death) for (dim, (birth, death)) in persistence if dim == 1]
    )


def transform_diagram(diagram: torch.Tensor):
    """Transform the diagram to a list of pi values (birth / death).

    Parameters
    ----------
    diagram : torch.Tensor
        Persistence diagram

    Returns
    -------
    torch.Tensor
        Tensor of pi values.
    """

    b, d = diagram[:, 0], diagram[:, 1]
    pi = d / b
    return np.log(pi)


def get_empirical_distribution(dim: int):
    """Generates empirical distribution of pi values for random pointcloud in R^{dim}

    Parameters
    ----------
    dim : int
        Dimension

    Returns
    -------
    ECDF
        CDF of the distribution.
    """
    random_pc = rng.uniform(size=(10000, dim))
    dgm_rand = persistent_homology(random_pc)
    return ECDF(transform_diagram(dgm_rand))


def test_weak_universality(emp_cdf: ECDF, diagram, alpha: float = 0.05):
    """Test cycles for significance using weak universality.
    See: Bobrowski, O., Skraba, P. A universal null-distribution for topological data analysis. Sci Rep 13, 12274 (2023).

    Parameters
    ----------
    emp_cdf : ECDF
        Emperical CDF of pi values of random points.
    diagram : _type_
        Persistence diagram
    alpha : float, optional
        p-value, by default 0.05

    Returns
    -------
    int
        Number of significant cycles.
    """
    pvals = 1 - emp_cdf(transform_diagram(diagram))
    is_significant, _, _, _ = mt.multipletests(pvals, alpha=alpha, method="bonferroni")
    return np.sum(is_significant)


def sample_points(points: torch.Tensor, n: int = 300):
    """Sample n random points.

    Parameters
    ----------
    points : torch.Tensor
        Points
    n : int, optional
        Size of sample, by default 300

    Returns
    -------
    torch.Tensor
        Sample
    """
    return points[rng.choice(points.shape[0], min(n, points.shape[0]), replace=False)]


class CoverLifting(PointCloud2GraphLifting):
    r"""Lifts point cloud data to graph by creating its k-NN graph

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class
    """

    def __init__(
        self,
        ambient_dim: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cover_complex = gudhi.cover_complex.MapperComplex(
            input_type="point cloud",
            min_points_per_node=0,
            clustering=None,
            N=100,
            beta=0.0,
            C=10,
            filter_bnds=None,
            resolutions=None,
            gains=None,
        )

        self.test_fn = partial(
            test_weak_universality, get_empirical_distribution(ambient_dim)
        )

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts a point cloud dataset to a graph by constructing its k-NN graph.

        Parameters
        ----------
        data :  torch_geometric.data.Data
            The input data to be lifted

        Returns
        -------
        dict
            The lifted topology
        """
        points = data.pos

        # use height function as the filter
        height = points[:, -1]
        _ = self.cover_complex.fit(data.pos, filters=height, colors=height)

        graph = self.cover_complex.get_networkx()

        removed_edges = []
        for u, v in graph.edges():
            u_inds = set([int(x) for x in self.cover_complex.node_info_[u]["indices"]])
            v_inds = set([int(x) for x in self.cover_complex.node_info_[v]["indices"]])

            interior = sample_points(points[list(u_inds & v_inds)])

            u_boundary = sample_points(points[list(u_inds - v_inds)])
            v_boundary = sample_points(points[list(v_inds - u_inds)])

            remove_edge = True
            if min(len(u_boundary), len(v_boundary)) == 0:
                remove_edge = False
            elif len(interior) > 0:
                # number of significant cycles
                num_cycles = self.test_fn(persistent_homology(interior))

                # number of significant relative cycles
                x = np.vstack([interior, u_boundary, v_boundary])
                num_relative_cycles = self.test_fn(
                    persistent_homology(
                        x, subcomplex_inds=np.arange(interior.shape[0], x.shape[0])
                    )
                )

                if num_relative_cycles > num_cycles:
                    remove_edge = False

            if remove_edge:
                removed_edges.append((u, v))

        graph.remove_edges_from(removed_edges)

        graph_data = from_networkx(graph)

        return {
            "num_nodes": graph_data.num_nodes,
            "edge_index": graph_data.edge_index,
            "x": torch.ones((graph_data.num_nodes, 1)),
        }
