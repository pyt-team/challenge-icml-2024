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
    """Matroids is a set system that implements the following

    Matroids are combinatorial structures that follow certain rules, see the accompanying functions:
    - from_bases
    - from_circuits
    - from_rank
    for more details.
    Moreover, if a set is independent:
    - (1) then its subsets are
    - (2) if there is a bigger independent set, we can construct a new nontrivial independent set by adding an element from the bigger set to the smaller.

    Parameters
    ----------
    ground : Collection
        A collection of ground elements that the matroid is based on. We need this to compute the dual matroid
    independent_sets : Collection, optional
        These are the simplicial complexes that follow (1) and (2).
        If None, then this is considered a dubious matroid, where only the empty set is independent.

    Examples
    --------
    Define the uniform matroid U24:

    >>> ground = range(4)
    >>> independent_sets = [subset for subset in power(ground) if len(subset) <= 2]
    >>> M = CCMatroid.from_bases(ground=ground, independent_sets=independent_sets)
    >>> M.skeleton(1) # [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    """

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
    def from_bases(cls, ground: Collection, bases: Collection | None) -> "CCMatroid":
        """Constructs a matroid from its bases
        The bases of a matroid are ones where we cannot use the augmentation axiom to further increase its cardinality.
        As matroids are simplicial complexes, we can use the independence criterion to form the independent sets.

        Parameters
        ----------
        ground : Collection
            A collection of ground elements that the matroid is based on.
        bases : Collection, optional
            A collection of bases

        Returns
        -------
        CCMatroid
            The associated matroid associated with the bases.
        """
        independent_sets = frozenset(
            [frozenset(ind) for base in bases for ind in powerset(base)]
        )
        return cls(ground=ground, independent_sets=independent_sets)

    @classmethod
    def from_circuits(cls, ground: Collection, circuits: Collection) -> "CCMatroid":
        """Constructs a matroid from its circuits.
        A set X is a circuit of M if every proper subset of X is independent in M, but X itself is not dependent.

        Parameters
        ----------
        ground : Collection
            A collection of ground elements that the matroid is based on.
        circuits : Collection, optional
            A collection of circuits

        Returns
        -------
        CCMatroid
            The associated matroid associated with the circuits.
        """
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
        """Constructs a matroid from its rank function
        A rank function is submodular, and we can use its definition to create the independent sets.
        More precisely, a subset X of the ground set is independent if its computed rank, r(X) == len(X)

        Parameters
        ----------
        ground : Collection
            A collection of ground elements that the matroid is based on.
        matroid_rank : Callable[[Collection], int]
            A rank function. A rank function is order preserving, submodular, and normalized.

        Returns
        -------
        CCMatroid
            The associated matroid associated with the rank function.
        """
        independent_sets = [
            subset
            for subset in powerset(groundset)
            if len(subset) == matroid_rank(subset)
        ]
        return cls(ground=groundset, independent_sets=independent_sets)

    def matroid_rank(self, input_set: Iterable) -> int:
        """Matroid rank function
        This is the original matroid rank function that isn't from the combinatorial complex rank.
        We need this function to compute certain matroids.
        It also has a nice definition with dual matroids.

        Parameters
        ----------
        input_set : Collection
            A collection of elements based on the current matroid's ground set.

        Returns
        -------
        int
            The computed matroid rank of the input.
        """
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
        raise NotImplementedError

    def span(self, S: Collection) -> frozenset:
        """Returns the closure of a set.
        The closure of a set $S$ is defined as the elements that do not contribute to rank of $S$.
        In some aspects, it represents $cls(S) - S$ represents the boundary of $S$ (much like normal topology).

        Parameters
        ----------
        S : Collection
            A collection of elements based on the current matroid's ground set.

        Returns
        -------
        frozenset
            The closure of S
        """
        S = fs(S)
        rankS = self.matroid_rank(S)
        closure = {
            g
            for g in (frozenset.difference(self.ground, S))
            if self.matroid_rank({g} | S) == rankS
        }
        return frozenset.union(S, frozenset(closure))

    def dual(self) -> "CCMatroid":
        """Returns the dual of a matroid
        The dual of a matroid is computed by taking the complement of its bases.

        Returns
        -------
        CCMatroid
            The dual of the associated matroid
        """
        new_bases = [frozenset.difference(self.ground, base) for base in self.bases]
        return CCMatroid.from_bases(ground=self.ground, bases=new_bases)

    def deletion(self, T: Collection) -> "CCMatroid":
        """Returns the deletion matroid of T
        The deletion of a matroid M on T is computed by removing T from all of its independent sets.

        Parameters
        ----------
        T : Collection
            A collection of elements to be removed based on the current matroid's ground set.

        Returns
        -------
        CCMatroid
            The deletion of the associated matroid
        """
        T = frozenset(T)
        new_bases = [base - T for base in self.bases]
        return CCMatroid.from_bases(ground=self.ground, bases=new_bases)

    def contraction(self, T: Collection) -> "CCMatroid":
        """Returns the contraction matroid of T
        The contraction of a matroid M on T is computed by removing T from all of its dual's independent sets.
        The explanation above is more apt explained akin to contraction of graphs, and
        thus we use an equivalence based on the dual matroid to compute easily.

        Parameters
        ----------
        T : Collection
            A collection of elements to be removed based on the current matroid's ground set.

        Returns
        -------
        CCMatroid
            The contraction of the associated matroid
        """
        T = frozenset(T)
        return CCMatroid.dual().deletion(T).dual()


class CCGraphicMatroid(CCMatroid):
    """Class for Graphic Matroids.

    A graphic matroid uses an underlying graph (edges) as the ground set of a matroid.
    Its bases are the spanning trees of the graph, which means the forests of a graph are the independent sets.

    Parameters
    ----------
    edgelist : Collection
        A collection of graph edges that the matroid is based on. Internally, it uses `nx.Graph` to compute the spanning trees.
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
        raise NotImplementedError
