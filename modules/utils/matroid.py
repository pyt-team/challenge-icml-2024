from itertools import chain, combinations
from typing import Iterable


def powerset(iterable: Iterable):
    """From https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    "powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class Matroid:
    """Matroids is a set system that implements the following"""

    def __init__(self, ground: Iterable, bases=Iterable[Iterable]):
        self._ground = frozenset(ground)
        self._bases = {frozenset(base) for base in bases}

        self._independent_sets = None
        self._circuits = None

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


if __name__ == "__main__":
    matroid = Matroid(
        {"a", "b", "c", "d"}, {frozenset({"a", "b"}), frozenset({"c", "d"})}
    )
    print(matroid.independent_sets)
    print(matroid.circuits)
