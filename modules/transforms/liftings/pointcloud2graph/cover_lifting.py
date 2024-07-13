from functools import partial

import gudhi
import gudhi.cover_complex
import numpy as np
import statsmodels.stats.multitest as mt
import torch
import torch_geometric
from gudhi import cover_complex
from statsmodels.distributions.empirical_distribution import ECDF
from torch_geometric.utils.convert import from_networkx

from modules.transforms.liftings.pointcloud2graph.base import PointCloud2GraphLifting


def persistent_homology(points: torch.Tensor, subcomplex_inds: list[int] = None):
    st = gudhi.AlphaComplex(points=points).create_simplex_tree()

    if subcomplex_inds is not None:
        subcomplex = []
        for simplex in st.get_simplices():
            if all(x in subcomplex_inds for x in simplex[0]):
                subcomplex.append(simplex[0])

        new_vertex = st.num_vertices()
        st.insert([new_vertex], 0)
        for simplex in subcomplex:
            st.insert(simplex + [new_vertex], st.filtration(simplex))

    persistence = st.persistence()
    diagram = np.array(
        [(birth, death) for (dim, (birth, death)) in persistence if dim == 1]
    )

    return diagram


def transform(diagram):
    b, d = diagram[:, 0], diagram[:, 1]
    pi = d / b
    return np.log(pi)


def get_empirical_distribution(dim: int):
    """Generates empirical distribution of pi values for random pointcloud in R^{dim}"""
    random_pc = np.random.uniform(size=(10000, dim))
    dgm_rand = persistent_homology(random_pc)
    return ECDF(transform(dgm_rand))


def test_weak_universality(emp_cdf: ECDF, diagram, alpha: float = 0.05):
    pvals = 1 - emp_cdf(transform(diagram))
    is_significant, _, _, _ = mt.multipletests(pvals, alpha=alpha, method="bonferroni")
    return np.sum(is_significant)


def sample_points(points: torch.Tensor, n=300):
    return points[
        np.random.choice(points.shape[0], min(n, points.shape[0]), replace=False)
    ]


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
        cover_complex: gudhi.cover_complex.CoverComplex = None,
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
