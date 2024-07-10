"""Test the message passing module."""

from math import comb

import torch
import torch_geometric
import torch_geometric.data

from modules.data.utils.utils import (
    load_double_house_graph,
    load_k4_graph,
    load_manual_graph,
)
from modules.matroid.matroid import CCMatroid, powerset
from modules.transforms.liftings.graph2combinatorial.base import (
    Matroid2CombinatorialLifting,
)
from modules.transforms.liftings.graph2combinatorial.curve_lifting import (
    GraphCurveMatroidLifting,
)


class TestGraphCurveLifting:
    """Test the CurveLifting class.

    It consists of 2 components: GraphCurveMatroidLifting & Matroid2CombinatorialLifting
    """

    def setup_method(self):
        # Load the graph
        self.data1 = load_manual_graph()

        self.graph_curve_lifting = GraphCurveMatroidLifting()
        self.matroid_lifting = Matroid2CombinatorialLifting()

    # Matroid Construction Test 1:
    def test_uniform_matroid_construction(self):
        """In this test, we test different ways of constructing matroids."""
        n = 11
        k = 4
        ground = frozenset(range(n))
        bases = frozenset([frozenset(s) for s in powerset(ground) if len(s) == k])

        bases_U_kn = CCMatroid.from_bases(ground=ground, bases=bases)
        for base in bases_U_kn.skeleton(k - 1):
            assert (
                len(base) == k
            ), f"Matroid construction of U^{n}_{k} gone wrong for {base}"

        circuits = frozenset(
            [frozenset(s) for s in powerset(ground) if len(s) == k + 1]
        )
        circuits_U_kn = CCMatroid.from_circuits(ground=ground, circuits=circuits)
        assert (
            circuits_U_kn.bases == bases_U_kn.bases
        ), "Uniform matroid constructed from bases and circuits not equal"

        def rank(X):
            nonlocal k
            return min(len(X), k)

        rank_U_kn = CCMatroid.from_rank(ground, rank)
        assert (
            circuits_U_kn.bases == rank_U_kn.bases
        ), "Uniform matroid constructed from circuits and rank not equal"

        assert (
            rank_U_kn.ground == ground
        ), "Uniform matroid ground set is constructed incorrectly."

    # Test 1: check for k4 matroid correctness
    def test_k4(self):
        """Test the graph curve matroid of the K_4 graph.
        In Geiger et al., it turns out the graph curve matroid on $K_4$ is actually isomorphic to the uniform matroid on 4 elements into 2 subsets.
        In order to prove correctness, we will show that the graph curve matroid of the K_4 graph is isomorphic to the U_2^4 matroid.
        """
        k4 = load_k4_graph()
        k4_curve = self.graph_curve_lifting.lift_topology(k4)

        k4_curve_matroid = CCMatroid.from_bases(
            ground=k4_curve["ground"], bases=k4_curve["bases"]
        )

        uniform_ground = k4_curve_matroid.ground
        uniform_bases = [s for s in powerset(uniform_ground) if len(s) == 2]

        u_24 = CCMatroid.from_bases(ground=uniform_ground, bases=uniform_bases)
        assert (
            u_24.bases == k4_curve_matroid.bases
        ), "Matroid creation is wrong, M_g(k4) is not isomorphic to U_2^4"

    # Test 2: check for double house matroid correctness
    def test_double_house_graph(self):
        """Test the graph curve matroid of the double house graph.
        In Geiger et al., the double house graph is important for calculation reasons.
        """
        double_house = load_double_house_graph()
        curve = self.graph_curve_lifting.lift_topology(double_house)
        double_house_matroid = CCMatroid.from_bases(
            ground=curve["ground"], bases=curve["bases"]
        )

        A = frozenset({0, 1, 2})

        assert (
            A not in double_house_matroid
        ), f"Graph Curve Matroid creation is wrong, {A} is not in the circuits of the double house graph"
        # [2, 3, 5, 6, 7] is a circuit
        assert [1, 2, 4, 5] in double_house_matroid.cells, "[1, 2, 4, 5, 6] circuit check fail"
        assert [1, 2, 4, 6] in double_house_matroid.cells, "[1, 2, 4, 5, 6] circuit check fail"
        assert [1, 2, 5, 6] in double_house_matroid.cells, "[1, 2, 4, 5, 6] circuit check fail"
        assert [1, 4, 5, 6] in double_house_matroid.cells, "[1, 2, 4, 5, 6] circuit check fail"
        assert [2, 4, 5, 6] in double_house_matroid.cells, "[1, 2, 4, 5, 6] circuit check fail"

        # rank check
        assert double_house_matroid.max_rank == 3 # this is 4 in matroid rank

        # bases check
        assert len(double_house_matroid.bases) == 54

    # Test 3: Test the matroid to combinatorial complex
    def test_lift_matroid2cc_topology(self):
        """We inspect the combinatorial complex of the uniform matroid.
        It's easy to count the level set of the uniform matroid, so we do some tests to do that.
        """
        n = 6
        k = 4

        uniform_ground = [i for i in range(n)]
        uniform_bases = [s for s in powerset(uniform_ground) if len(s) == k]

        u_kn = CCMatroid.from_bases(ground=uniform_ground, bases=uniform_bases)

        cc = self.matroid_lifting.matroid2cc(u_kn)
        assert len(cc.skeleton(1 - 1)) == comb(
            n, 1
        ), f"Matroid 2 combinatorial complex was generated incorrectly for rank={1 - 1}"
        assert len(cc.skeleton(2 - 1)) == comb(
            n, 2
        ), f"Matroid 2 combinatorial complex was generated incorrectly for rank={2 - 1}"
        assert len(cc.skeleton(3 - 1)) == comb(
            n, 3
        ), f"Matroid 2 combinatorial complex was generated incorrectly for rank={3 - 1}"
        assert len(cc.skeleton(4 - 1)) == comb(
            n, 4
        ), f"Matroid 2 combinatorial complex was generated incorrectly for rank={4 - 1}"

    # Test 4
    def test_lift_topology(self):
        """Test the actual incidence matrix of the matroid lift"""
        n = 6
        k = 4

        uniform_ground = [i for i in range(n)]
        uniform_bases = [s for s in powerset(uniform_ground) if len(s) == k]

        features = torch.ones((n, 1))
        data = torch_geometric.data.Data(
            x=features, ground=uniform_ground, bases=uniform_bases
        )

        lifted_data = self.matroid_lifting.forward(data)
        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )

        assert (
            expected_incidence_1 == lifted_data["incidence_1"].to_dense()
        ).all(), "Something is wrong with incidence_1."

    # Test 5
    def test_lift_topology_rank(self):
        """Test the actual incidence matrix of the matroid lift wrt to truncation"""
        n = 6
        k = 4

        matroid_lifting_k1 = Matroid2CombinatorialLifting(max_rank=k - 2)
        assert k - 2 > 0
        uniform_ground = [i for i in range(n)]
        uniform_bases = [s for s in powerset(uniform_ground) if len(s) == k]

        features = torch.ones((n, 1))
        data = torch_geometric.data.Data(
            x=features, ground=uniform_ground, bases=uniform_bases
        )

        lifted_data = matroid_lifting_k1.forward(data)
        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )


        assert (
            expected_incidence_1 == lifted_data["incidence_1"].to_dense()
        ).all(), "Something is wrong with incidence_1. The lift under a truncated matroid is not the same."
