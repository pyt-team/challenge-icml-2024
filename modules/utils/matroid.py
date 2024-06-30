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
        self.circuits = None

    def independent_sets(self) -> frozenset:
        if self._independent_sets:
            return self._independent_sets
        self._independent_sets = frozenset(
            {frozenset(ind) for base in self._bases for ind in powerset(base)}
        )
        return self._independent_sets


if __name__ == "__main__":
    matroid = Matroid(
        {"a", "b", "c", "d"}, {frozenset({"a", "b"}), frozenset({"c", "d"})}
    )
    print(matroid.independent_sets())
