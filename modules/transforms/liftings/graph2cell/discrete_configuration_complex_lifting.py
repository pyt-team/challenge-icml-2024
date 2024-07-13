from itertools import permutations
from typing import Tuple

import networkx as nx
import torch_geometric
from toponetx.classes import CellComplex

from modules.transforms.liftings.graph2cell.base import Graph2CellLifting

Vertex = int
Edge = Tuple[Vertex, Vertex]
ConfigurationTuple = Tuple[Vertex | Edge]


def DiscreteConfigurationComplex(k: int, graph: nx.Graph):
    class Configuration:
        _instances: dict[ConfigurationTuple, "Configuration"] = {}
        _counter = 0

        def __new__(cls, configuration_tuple: ConfigurationTuple):
            # Create a key from the arguments
            key = configuration

            # If an instance doesn't exist for these arguments, create one
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)

            # Return the instance for these arguments
            return cls._instances[key]

        def __init__(self, configuration_tuple: ConfigurationTuple) -> None:
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
                self.contents = {Configuration._counter}
                Configuration._counter += 1
            else:
                self.contents = set()

            self._upwards_neighbors_generated = False
            self._generate_upwards_neighbors()

        def _generate_upwards_neighbors(self):
            if self._upwards_neighbors_generated:
                return
            self._upwards_neighbors_generated = True
            for i, agent in enumerate(self.configuration_tuple):
                if isinstance(agent, Edge):
                    continue
                for neighbor in graph[agent]:
                    self._generate_new_configuration(i, agent, neighbor)

        def _generate_new_configuration(
            self, index: int, vertex_agent: Vertex, neighbor: Vertex
        ):
            if neighbor in self.neighborhood:
                return
            new_edge = (min(vertex_agent, neighbor), max(vertex_agent, neighbor))
            new_configuration_tuple = (
                *self.configuration_tuple[:index],
                new_edge,
                *self.configuration_tuple[index + 1 :],
            )
            new_configuration = Configuration(new_configuration_tuple)
            new_configuration.contents.add(frozenset(self.contents))
            new_configuration._generate_upwards_neighbors()

    for dim_0_configuration_tuple in permutations(graph, k):
        configuration = Configuration(dim_0_configuration_tuple)

    cells = {i: [] for i in range(k + 1)}
    for conf in Configuration._instances.values():
        cells[conf.dim].append(conf.contents)

    return cells


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
        if k < 0 or k > 2:
            raise NotImplementedError(
                "Only k = 0, 1, or 2 is currently supported. This is due to TopoNetX only supporting cell complexes of dimension 2. This may change in the future."
            )
        super().__init__(**kwargs)
        self.k = k

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
        G = self._generate_graph_from_data(data)
        cell_complex_data = DiscreteConfigurationComplex(self.k, G)
        cc = CellComplex.from_networkx_graph()
        cc.from_networkx_graph(nx.Graph(cell_complex_data[1]))

        if self.k == 1:
            return

        for dim_2_cell in cell_complex_data[2]:
            cycle = [e[0] for e in nx.find_cycle(nx.Graph(dim_2_cell))]
            cc.add_cell(cell=cycle, rank=2)

        return
