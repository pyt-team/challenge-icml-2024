"""Test the path lifting module."""

import numpy as np

from modules.data.load.loaders import GraphLoader
from modules.transforms.liftings.graph2hypergraph.path_lifting import PathLifting
from modules.utils.utils import load_dataset_config


class TestHypergraphPathLifting:
    """Test the PathLifting class."""

    def setup_method(self):
        """Initialise the PathLifting class."""
        dataset_config = load_dataset_config("manual_dataset")
        loader = GraphLoader(dataset_config)
        self.dataset = loader.load()
        self.data = self.dataset._data

    def test_true(self):
        """Naive test to check if the test is running."""
        assert True

    # def test_false(self):
    #     """Naive test to check if the test is running."""
    #     assert False

    def test_1(self):
        """Verifies setup_method is working."""
        assert self.dataset is not None

    def test_2(self):
        """test: no target node for one source node returns something"""
        source_nodes = [0, 2]
        target_nodes = [1, None]
        lengths = [2, 2]
        include_smaller_paths = True
        path_lifting = PathLifting(
            source_nodes,
            target_nodes,
            lengths,
            include_smaller_paths=include_smaller_paths,
        )
        res = path_lifting.find_hyperedges(self.data)
        res_expected = [
            [0, 1],
            [0, 1, 2],
            [0, 4, 1],
            [2, 4],
            [2, 1],
            [2, 0],
            [2, 7],
            [2, 5],
            [2, 3],
            [2, 1, 4],
            [2, 4, 0],
            [2, 1, 0],
            [2, 0, 7],
            [2, 5, 7],
            [2, 3, 6],
            [2, 5, 6],
            # [],
        ]
        assert {frozenset(x) for x in res_expected} == res

    def test_3(self):
        """test: include_smaller_paths=False"""
        source_nodes = [0]
        target_nodes = [1]
        lengths = [2]
        include_smaller_paths = False
        res = PathLifting(
            source_nodes,
            target_nodes,
            lengths,
            include_smaller_paths=include_smaller_paths,
        ).find_hyperedges(self.data)
        assert frozenset({0, 1}) not in res

    def test_4(self):
        """test: include_smaller_paths=True"""
        source_nodes = [0]
        target_nodes = [1]
        lengths = [2]
        include_smaller_paths = True
        res = PathLifting(
            source_nodes,
            target_nodes,
            lengths,
            include_smaller_paths=include_smaller_paths,
        ).find_hyperedges(self.data)
        assert frozenset({0, 1}) in res

    def test_5(self):
        """test: when include_smaller_paths=False all paths have the length specified"""
        source_nodes = [0]
        target_nodes = [1]
        include_smaller_paths = False
        for k in range(1, 5):
            lengths = [k]
            res = PathLifting(
                source_nodes,
                target_nodes,
                lengths,
                include_smaller_paths=include_smaller_paths,
            ).find_hyperedges(self.data)
            assert np.array([len(x) - 1 == k for x in res]).all()

    def test_6(self):
        """test: no target node global returns something"""
        source_nodes = [0, 1]
        target_nodes = None
        lengths = [2, 2]
        include_smaller_paths = False
        res = PathLifting(
            source_nodes,
            target_nodes,
            lengths,
            include_smaller_paths=include_smaller_paths,
        ).find_hyperedges(self.data)
        assert len(res) > 0

    def test_7(self):
        """test: every hyperedge contains the source and target nodes when specified"""
        a = np.random.default_rng().choice(
            np.arange(len(self.data.x)), 2, replace=False
        )
        source_nodes = [a[0]]
        target_nodes = [a[1]]
        lengths = [np.random.default_rng().integers(1, 5)]
        include_smaller_paths = False
        res = PathLifting(
            source_nodes,
            target_nodes,
            lengths,
            include_smaller_paths=include_smaller_paths,
        ).find_hyperedges(self.data)
        if len(res) > 0:
            assert (
                np.array([source_nodes[0] in x for x in res]).all()
                and np.array([target_nodes[0] in x for x in res]).all()
            )

    def test_8(self):
        """test: no target node for one source node returns something"""
        source_nodes = [0, 2]
        target_nodes = [1, None]
        lengths = [2, 2]
        include_smaller_paths = False
        res = PathLifting(
            source_nodes,
            target_nodes,
            lengths,
            include_smaller_paths=include_smaller_paths,
        ).find_hyperedges(self.data)
        assert len(res) > 0
