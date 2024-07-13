from itertools import permutations
from typing import ClassVar

import networkx as nx
import torch_geometric
from toponetx.classes import CellComplex

from modules.transforms.liftings.graph2cell.base import Graph2CellLifting
from modules.utils.utils import edge_cycle_to_vertex_cycle

Vertex = int
Edge = tuple[Vertex, Vertex]
ConfigurationTuple = tuple[Vertex | Edge]


class DiscreteConfigurationLifting(Graph2CellLifting):
    r"""Lifts graphs to cell complexes by generating the k-th *discrete configuration complex* $D_k(G)$ of the graph. This is a cube complex, which is similar to a simplicial complex except each n-dimensional cell is homeomorphic to a n-dimensional cube rather than an n-dimensional simplex.

    The discrete configuration complex of order k consists of all sets of k unique edges or vertices of $G$, with the additional constraint that if an edge e is in a cell, then neither of the endpoints of e are in the cell. For examples of different graphs and their configuration complexes, see the tutorial.

    Parameters
    ----------
    k : int, optional.
        The order of the configuration complex, or the number of 'points' in the configuration. Currently only k <= 2 is supported, since TopoNetX only allows cell complexes to have dimension 2. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, k: int = 2, **kwargs):
        if k < 1 or k > 2:
            raise NotImplementedError(
                "Only k = 1, or 2 is currently supported. This is due to TopoNetX only supporting cell complexes of dimension 2. This may change in the future."
            )
        super().__init__(**kwargs)
        self.k = k

    def _get_lifted_topology(self, cell_complex: CellComplex) -> dict:
        raise NotImplementedError()

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Generates the cubical complex of discrete graph configurations.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        # configurations of k = 1 agents are the same as the graph
        if self.k == 1:
            return data

        G = self._generate_graph_from_data(data)
        if G.is_directed():
            raise ValueError("Directed Graphs are not supported.")

        cell_complex_data = generate_configuration_complex_data(self.k, G)
        cc = CellComplex()
        cc.from_networkx_graph(nx.Graph(cell_complex_data[1]))

        if self.k == 1:
            return None

        for dim_2_cell in cell_complex_data[2]:
            cc.add_cell(edge_cycle_to_vertex_cycle(dim_2_cell), rank=2)

        return self._get_lifted_topology(cc)


def generate_configuration_class(k: int, graph: nx.Graph):
    """Class factory for the Configuration class."""

    class Configuration:
        """Represents a single legal configuration of k agents on a graph G. A legal configuration is a tuple of k edges and vertices of G where all the vertices and endpoints are **distinct** i.e. no two edges sharing an endpoint can simultaneously be in the configuration, and adjacent (edge, vertex) pair can be contained in the configuration.

        Parameters
        ----------
        k : int, optional.
            The order of the configuration complex, or the number of 'points' in the configuration.
        graph: nx.Graph.
            The graph on which the configurations are defined.
        """

        instances: ClassVar[dict[ConfigurationTuple, "Configuration"]] = {}
        _counter = 0

        def __new__(cls, configuration_tuple: ConfigurationTuple):
            # Ensure that a configuration tuple corresponds to a *unique* configuration object
            key = configuration_tuple
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)

            return cls._instances[key]

        def __init__(self, configuration_tuple: ConfigurationTuple) -> None:
            # If this object was already initialized earlier, maintain current state
            if hasattr(self, "initialized"):
                return

            self.initialized = True
            self.configuration_tuple = configuration_tuple
            self.neighborhood = set()
            self.dim = 0
            for agent in configuration_tuple:
                if isinstance(agent, Edge):
                    self.neighborhood.update(set(agent))
                    self.dim += 1
                else:
                    self.neighborhood.add(agent)

            if self.dim == 0:
                self.contents = Configuration._counter
                Configuration._counter += 1
            else:
                self.contents = []

            self._upwards_neighbors_generated = False

        def generate_upwards_neighbors(self):
            """For the configuration self of dimension d, generate the configurations of dimension d+1 containing it."""
            if self._upwards_neighbors_generated:
                return
            self._upwards_neighbors_generated = True
            for i, agent in enumerate(self.configuration_tuple):
                if isinstance(agent, tuple):
                    continue
                for neighbor in graph[agent]:
                    self._generate_single_neighbor(i, agent, neighbor)

        def _generate_single_neighbor(
            self, index: int, vertex_agent: int, neighbor: int
        ):
            """Generate a configuration containing the configuration self by 'expanding' an edge."""
            # If adding the edge (vertex_agent, neighbor) would produce an illegal configuration, ignore it
            if neighbor in self.neighborhood:
                return

            # We always orient edges (min -> max) to maintain uniqueness of configuration tuples
            new_edge = (min(vertex_agent, neighbor), max(vertex_agent, neighbor))
            new_configuration_tuple = (
                *self.configuration_tuple[:index],
                new_edge,
                *self.configuration_tuple[index + 1 :],
            )
            new_configuration = Configuration(new_configuration_tuple)
            new_configuration.contents.append(self.contents)
            new_configuration.generate_upwards_neighbors()

    return Configuration


def generate_configuration_complex_data(k: int, graph: nx.Graph):
    """Generate the cell data of the configuration complex $D_k(G)$."""
    Configuration = generate_configuration_class(k, graph)

    # The vertices of the configuration complex are just tuples of k vertices
    for dim_0_configuration_tuple in permutations(graph, k):
        configuration = Configuration(dim_0_configuration_tuple)
        configuration.generate_upwards_neighbors()

    cells = {i: [] for i in range(k + 1)}
    for conf in Configuration.instances.values():
        cells[conf.dim].append(conf.contents)

    return cells
