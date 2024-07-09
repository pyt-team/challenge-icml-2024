from collections.abc import Callable, Collection, Iterable
from itertools import chain, combinations

import networkx as nx
from toponetx.classes.combinatorial_complex import CombinatorialComplex


def powerset(iterable: Iterable):
    """From https://docs.python.org/3/library/itertools.html#itertools-recipes
    Example: powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def fs(iterable: Iterable):
    return iterable if isinstance(iterable, frozenset) else frozenset(iterable)


class CCMatroid(CombinatorialComplex):
    """Matroids is a set system that implements the following"""

    def __init__(
        self,
        ground: Collection,
        independent_sets: Collection | None = None,
        **kwargs,
    ) -> None:
        CombinatorialComplex.__init__(
            self, cells=None, ranks=None, graph_based=False, **kwargs
        )
        # Remove the empty set from the set of independent sets.
        independent_sets = list(filter(lambda X: len(X) != 0, independent_sets))
        ranks = map(lambda X: len(X) - 1, independent_sets)

        for ind, rnk in zip(independent_sets, ranks, strict=True):
            self.add_cell(ind, rnk)
        self.max_rank = self.ranks[-1]
        self._ground = frozenset([g for g in ground])

    @classmethod
    def from_bases(cls, ground: Collection, bases: Collection) -> "CCMatroid":
        independent_sets = frozenset(
            [frozenset(ind) for base in bases for ind in powerset(base)]
        )
        return cls(ground=ground, independent_sets=independent_sets)

    @classmethod
    def from_circuits(cls, ground: Collection, circuits: Collection) -> "CCMatroid":
        """A set X is a circuit of M if every proper subset of X is independent in M, but X itself is not dependent."""
        independent_sets = []
        for circuit in circuits:
            for ind in powerset(circuit):
                ind = frozenset(ind)
                if ind == circuit:
                    continue
                independent_sets.append(ind)
        return cls(ground=ground, independent_sets=independent_sets)

    @classmethod
    def from_rank(
        cls, groundset: Collection, matroid_rank: Callable[[Collection], int]
    ) -> "CCMatroid":
        independent_sets = [
            subset
            for subset in powerset(groundset)
            if len(subset) == matroid_rank(subset)
        ]
        return cls(ground=groundset, independent_sets=independent_sets)

    def matroid_rank(self, input_set: Iterable) -> int:
        input_set = fs(input_set)
        size = len(input_set)

        for level in range(size - 1, -1, -1):
            skeleton = self.skeleton(level)
            for independent_set in skeleton:
                if frozenset.issubset(independent_set, input_set):
                    return level + 1

        raise KeyError(f"hyperedge {input_set} and its subsets are not in the complex")

    @property
    def ground(self) -> frozenset:
        return self._ground

    @property
    def bases(self) -> frozenset:
        return frozenset(self.skeleton(self.max_rank))

    @property
    def circuits(self) -> Collection:
        """Not implemented; not important to implement for the purposes of the PR"""

    def span(self, S: Collection) -> frozenset:
        S = fs(S)
        rankS = self.matroid_rank(S)
        closure = {
            g
            for g in (frozenset.difference(self.ground, S))
            if self.matroid_rank({g} | S) == rankS
        }
        return frozenset.union(S, frozenset(closure))

    def dual(self) -> "CCMatroid":
        new_bases = [frozenset.difference(self.ground, base) for base in self.bases]
        return CCMatroid.from_bases(ground=self.ground, bases=new_bases)

    def deletion(self, T: Collection) -> "CCMatroid":
        T = frozenset(T)
        new_bases = [base - T for base in self.bases]
        return CCMatroid.from_bases(ground=self.ground, bases=new_bases)

    def contraction(self, T: Collection) -> "CCMatroid":
        T = frozenset(T)
        return CCMatroid.dual().deletion(T).dual()


class CCGraphicMatroid(CCMatroid):
    """Class for Graphic Matroids.

    A graphic matroid uses an underlying graph (edges) as the ground set of a matroid.
    Its bases are the spanning trees of the graph, which means the forests of a graph are the independent sets.
    """

    def __init__(self, edgelist: Collection, **kwargs):
        graph = nx.Graph()
        graph.add_edges_from(edgelist)
        ground_edges = [tuple(edge) for edge in graph.edges]
        spanning_trees = []
        for tree in nx.SpanningTreeIterator(graph):
            edges = tree.edges()
            cvt_tree = [tuple(edge) for edge in edges]
            spanning_trees.append(frozenset(cvt_tree))
        independent_sets = frozenset(
            [frozenset(ind) for base in spanning_trees for ind in powerset(base)]
        )
        CCMatroid.__init__(self, ground=ground_edges, independent_sets=independent_sets)

    def graph(self) -> nx.Graph:
        """Create the graph from the 0th skeleton. Use some kind of contraction mapping."""
