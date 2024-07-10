from collections import defaultdict
from typing import Any, Optional

import networkx as nx
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import CombinatorialComplex

from modules.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


def get_all_paths_from_single_node(
    n: int, g: dict[int, list[int]], path_length: int
) -> set[frozenset[int]]:
    r"""Get all paths from a dictionary of edges and a list of nodes

    Parameters
    ----------
    n : int
        Node to start the paths from.
    g : Dict[int, List[int]]
        Graph.
    path_length : int
        Length of the paths.

    Returns
    -------
        Set[FrozenSet[int]]:
            List of paths.
    """
    paths = set()
    if path_length == 1:
        paths.add(frozenset([n]))
        return paths
    for v in g[n]:
        sub_paths = get_all_paths_from_single_node(v, g, path_length - 1)
        for path in sub_paths:
            if n not in path:  # the path will be of length path_length without cycles
                new_path = frozenset([n]) | path
                paths.add(new_path)
    return paths


def get_all_paths_from_nodes(
    nodes: list[int], g: dict[int, list[int]], path_length: int
) -> set[frozenset[int]]:
    r"""Get all paths from a dictionary of edges and a list of nodes

    Parameters
    ----------
    nodes : List[int]
        List of nodes to start the paths from.
    g : Dict[int, List[int]]
        Graph.
    path_length : int
        Length of the paths.

    Returns
    -------
        Set[FrozenSet[int]]
            List of paths
    """
    paths = set()
    for n in nodes:
        if n in g:
            n_paths = get_all_paths_from_single_node(n, g, path_length)
            for path in n_paths:
                paths.add(path)
    return paths


class CombinatorialPathLifting(Graph2CombinatorialLifting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def path_based_lift_CC(
        self, input_cc: CombinatorialComplex, sources_nodes: list[int], path_length: int
    ) -> CombinatorialComplex:
        r"""Lift a 1-dimensional CC to a higher-dimensional CC by lifting the paths to higher-rank cells.

        Parameters
        ----------
        input_cc : CombinatorialComplex
            Original combinatorial complex.
        sources_nodes : List[int]
            List of source nodes to start the paths from.
        path_length : int
            Length of the paths to lift.

        Returns
        -------
            CombinatorialComplex
                Lifted combinatorial complex.
        """
        # Copy the rank-0 and rank-1 cells from the input CC
        cc = CombinatorialComplex()
        for rank in input_cc.cells.hyperedge_dict:
            cells = input_cc.cells.hyperedge_dict[rank]
            for cell in cells:
                attr = cells[cell]
                cc.add_cell(cell, rank=rank, **attr)

        # Add the paths as higher-rank cells
        edges = input_cc.cells.hyperedge_dict[1]
        graph = defaultdict(list)
        for e in edges:
            edge = tuple(e)
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        paths = get_all_paths_from_nodes(sources_nodes, graph, path_length)
        for path in paths:
            if len(path) == path_length:  # Ensure correct path length
                cc.add_cell(path, rank=2)  # Add paths as rank-2 cells
            # cc.add_cell(path, rank=2)  # Add paths as rank-2 cells
        return cc

    def graph_to_CCs(
        self,
        graphs: list[nx.Graph],
        lifting_procedure: Optional[str] = None,
        lifting_procedure_kwargs: str | dict[Any, Any] = None,
        **kwargs,
    ) -> list[CombinatorialComplex]:
        r"""Convert a list of graphs to a list of combinatorial complexes (of dimension 1).

        Parameters
        ----------
        graphs : List[nx.Graph]
            List of graphs.
        is_molecule : bool, optional
            Whether the graphs are molecules. Defaults to False.
        lifting_procedure : Optional[str], optional
            Lifting procedure to use. Defaults to None.
        lifting_procedure_kwargs : Optional[Union[str, Dict[Any, Any]]], optional
            Kwargs for the lifting procedure. Defaults to None.

        Returns
        -------
        List
            List of combinatorial complexes.
        """
        ccs = []
        for graph in graphs:
            CC = CombinatorialComplex()
            for node in graph.nodes:
                attr = graph.nodes[node]
                CC.add_cell((node,), rank=0, **attr)
            for edge in graph.edges:
                attr = graph.edges[edge]
                CC.add_cell(edge, rank=1, **attr)

            # Implement path based lift procedure
            if lifting_procedure is not None:  # lift to higher order
                if lifting_procedure_kwargs is None:
                    lifting_procedure_kwargs = {}
                if lifting_procedure == "path_based":
                    if isinstance(lifting_procedure_kwargs, str):
                        if lifting_procedure_kwargs == "basic":
                            max_nb_nodes = kwargs.get(
                                "max_nb_nodes",
                                max([g.number_of_nodes() for g in graphs]),
                            )
                            lifting_procedure_kwargs = {
                                "sources_nodes": list(range(max_nb_nodes)),
                                "path_length": 3,
                            }
                        else:
                            raise NotImplementedError(
                                f"Lifting procedure kwargs {lifting_procedure_kwargs} not implemented"
                            )
                    CC = self.path_based_lift_CC(CC, **lifting_procedure_kwargs)
                else:
                    raise NotImplementedError(
                        f"Lifting procedure {lifting_procedure} not implemented"
                    )
            ccs.append(CC)
        return ccs

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts a graph to a combinatorial path complex topology.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        # Convert torch_geometric data to networkx graph
        data = data.clone()
        graph_data = self._generate_graph_from_data(data)

        # Create a combinatorial complex from the graph
        ccs = self.graph_to_CCs(
            [graph_data],
            lifting_procedure="path_based",
            lifting_procedure_kwargs="basic",
        )
        lifting_topology = {}
        for cc in ccs:
            # Extract node features
            lifting_topology["x_0"] = data["x"]

            # Compute incidence hyperedges
            hyperedges = list(cc.cells.hyperedge_dict.get(2, {}).keys())

            num_nodes = data.num_nodes
            incidence_matrix = torch.zeros((num_nodes, len(hyperedges)))
            for i, hyper_edge in enumerate(hyperedges):
                for node in hyper_edge:
                    incidence_matrix[node, i] = 1
            lifting_topology["incidence_hyperedges"] = torch.Tensor(
                incidence_matrix
            ).to_sparse_coo()

            # Compute adjacency and incidence matrices
            for r in range(cc.dim + 1):
                if r < cc.dim:
                    lifting_topology[f"incidence_{r + 1}"] = from_sparse(
                        cc.incidence_matrix(r, r + 1, incidence_type="up")
                    )
                lifting_topology[f"adjacency_{r}"] = from_sparse(
                    cc.adjacency_matrix(r, r + 1)
                )

            # Compute number of hyper-edges
            lifting_topology["num_hyperedges"] = len(hyperedges)
        return lifting_topology
