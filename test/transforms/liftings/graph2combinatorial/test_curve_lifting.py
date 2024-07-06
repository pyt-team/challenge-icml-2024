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
from modules.transforms.liftings.graph2combinatorial.base import (
    Matroid2CombinatorialLifting,
)
from modules.transforms.liftings.graph2combinatorial.curve_lifting import (
    GraphCurveMatroidLifting,
)
from modules.utils.matroid import Matroid, powerset


class TestGraphCurveLifting:
    """Test the CurveLifting class.
    It consists of 2 components: GraphCurveMatroidLifting & Matroid2CombinatorialLifting
    In order to prove correctness, we will show that the graph curve matroid of the K_4 graph is isomorphic to the U_2^4 matroid.
    """

    def setup_method(self):
        # Load the graph
        self.data1 = load_manual_graph()

        self.graph_curve_lifting = GraphCurveMatroidLifting()
        self.matroid_lifting = Matroid2CombinatorialLifting()

    # Test 1: check for k4 matroid correctness
    def test_k4(self):
        """Test the graph curve matroid of the K_4 graph.
        In Geiger et al., it turns out the graph curve matroid on $K_4$ is actually isomorphic to the uniform matroid on 4 elements into 2 subsets.
        """
        k4 = load_k4_graph()
        k4_curve = self.graph_curve_lifting.lift_topology(k4)

        uniform_ground = k4_curve["ground"]
        uniform_bases = [s for s in powerset(uniform_ground) if len(s) == 2]

        u_24 = Matroid(ground=uniform_ground, bases=uniform_bases)
        k4_curve_matroid = Matroid(ground=k4_curve["ground"], bases=k4_curve["bases"])
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
        double_house_matroid = Matroid(ground=curve["ground"], bases=curve["bases"])

        A = frozenset({0, 1, 2})

        assert (
            A in double_house_matroid.circuits
        ), f"Graph Curve Matroid creation is wrong, {A} is not in the circuits of the double house graph"

    # Test 3: Test the matroid to combinatorial complex
    def test_lift_matroid2cc_topology(self):
        """We inspect the combinatorial complex of the uniform matroid."""
        n = 6
        k = 4

        uniform_ground = [i for i in range(n)]
        uniform_bases = [s for s in powerset(uniform_ground) if len(s) == k]

        u_kn = Matroid(ground=uniform_ground, bases=uniform_bases)

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
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
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

        matroid_lifting_k1 = Matroid2CombinatorialLifting(max_rank=k-2)
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
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            ]
        )

        assert (
            expected_incidence_1 == lifted_data["incidence_1"].to_dense()
        ).all(), "Something is wrong with incidence_1. The lift under a truncated matroid is not the same."
