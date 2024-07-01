import copy
from itertools import chain, combinations
from typing import Callable, Dict, Iterable

import matplotlib.pyplot as plt
import networkx as nx


def powerset(iterable: Iterable):
    """From https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    "powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def fs(iterable: Iterable):
    return iterable if isinstance(iterable, frozenset) else frozenset(iterable)


class Matroid:
    """Matroids is a set system that implements the following"""

    def __init__(self, ground: Iterable, bases=Iterable[Iterable]):
        self._ground = fs(ground)
        self._bases = fs([fs(base) for base in bases])

        self._independent_sets = None
        self._circuits = None

    @classmethod
    def create(cls, ground: Iterable, bases=Iterable[Iterable]):
        return cls(ground, bases)

    @property
    def bases(self) -> frozenset:
        return self._bases

    @property
    def independent_sets(self) -> frozenset:
        if self._independent_sets:
            return self._independent_sets
        self._independent_sets = frozenset(
            [frozenset(ind) for base in self._bases for ind in powerset(base)]
        )
        return self._independent_sets

    @property
    def circuits(self) -> frozenset:
        """A set X is a circuit of M if every proper subset of X is independent in M, but X itself is not dependent
        The computation for this is not very good, and more efficient implementations exist.
        """
        if self._circuits:
            return self._circuits
        circuits = []
        for subset in powerset(self._ground):
            if len(subset) == 0:  # skip the empty set
                continue
            subset = frozenset(subset)
            is_all_ind = (
                True  # tracks if the subset contains all independet proper subsets
            )

            for proper in powerset(subset):
                proper = frozenset(proper)
                if proper == subset:
                    continue
                if proper not in self.independent_sets:
                    is_all_ind = False
                    break
            if is_all_ind and subset not in self.independent_sets:
                circuits.append(subset)
        self._circuits = frozenset(circuits)
        return self._circuits

    def _rank(self, input_set: Iterable) -> int:
        input_set = fs(input_set)
        max_rank = 0
        for subset in powerset(input_set):
            if frozenset(subset) in self.independent_sets:
                max_rank = max(len(subset), max_rank)
        return max_rank

    @property
    def rank(self) -> Callable[[Iterable], int]:
        return self._rank

    def span(self, S: Iterable):
        S = fs(S)
        rankS = self._rank(S)
        return S | frozenset(
            {g for g in (self._ground - S) if self._rank({g} | S) == rankS}
        )

    def dual(self):
        return Matroid(
            ground=self._ground, bases=[self._ground - base for base in self.bases]
        )

    def deletion(self, T: frozenset):
        new_ground = self._ground - T
        # compute bases
        new_ind = [ind - T for ind in self.independent_sets]
        new_ind.sort(key=lambda s: -len(s))

        new_rank = len(new_ind[0])
        new_bases = frozenset(
            [frozenset(ind) for ind in new_ind if len(ind) == new_rank]
        )
        return Matroid(ground=new_ground, bases=new_bases)

    def contraction(self, T: frozenset):
        return self.dual().deletion(T).dual()


class GraphicMatroid(Matroid):
    """A graphic matroid uses an underlying graph (edges) as the ground set of a matroid. Its bases are the spanning trees of the graph, which means the forests of a graph are the independent sets."""

    def __init__(self, edgelist: Iterable, contract_map: Dict = None):
        graph = nx.Graph()
        graph.add_edges_from(edgelist)
        ground_edges = []
        for edge in graph.edges:
            ground_edges.append(tuple(edge))
        spanning_trees = []
        for tree in nx.SpanningTreeIterator(graph):
            edges = tree.edges()
            cvt_tree = []
            for edge in edges:
                cvt_tree.append(tuple(edge))
            spanning_trees.append(frozenset(cvt_tree))

        edges = frozenset(ground_edges)
        super(GraphicMatroid, self).__init__(ground=edges, bases=spanning_trees)
        self.vertices = []
        self.edges = []
        self.contract_map = contract_map if contract_map else {}
        for v_1, v_2 in edges:
            n1 = ",".join(self.contract_map.get(v_1, [v_1]))
            n2 = ",".join(self.contract_map.get(v_2, [v_2]))
            if n1 == n2:  # avoid self loops
                continue
            self.vertices.append(n1)
            self.vertices.append(n2)

            self.edges.append((n1, n2, f"{v_1},{v_2}"))
        self.vertices = frozenset(self.vertices)

    def graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        G.add_edges_from(self.edges)
        return G

    def deletion(self, T: frozenset):
        new_edges = super().deletion(T)._ground
        return GraphicMatroid(edgelist=new_edges)


if __name__ == "__main__":
    # ground = [c for c in "abcd"]
    # bases = ["ab", "ac", "ad", "bc", "bd", "cd"]
    # bases = [[c for c in base] for base in bases]
    # matroid = Matroid(ground, bases)
    # print(matroid.bases)
    # print("new bases")
    # print(matroid.deletion(frozenset(["a"])).bases)
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A"), ("B", "D")]
    M = GraphicMatroid(edgelist=edges)
    print(M.bases)
    # nx.draw(M.graph())
    # plt.show()

    # M = M.contraction(frozenset({("B", "D")}))
    # nx.draw(M.graph())
    # print(M.bases)
    # plt.show()
