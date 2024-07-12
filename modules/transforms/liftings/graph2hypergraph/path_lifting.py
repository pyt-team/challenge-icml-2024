"""A module for the PathLifting class."""
import networkx as nx
import numpy as np
import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class PathLifting(Graph2HypergraphLifting):
    """Lifts graphs to hypergraph domain by considering paths between nodes."""

    def __init__(
        self,
        source_nodes: list[int] = None,
        target_nodes: list[int] = None,
        lengths: list[int] = None,
        include_smaller_paths=False,
        **kwargs,
    ):
        # guard clauses
        if (
            lengths is not None
            and source_nodes is not None
            and len(source_nodes) != len(lengths)
        ):
            raise ValueError("source_nodes and lengths must have the same length")
        if target_nodes is not None and len(target_nodes) != len(source_nodes):
            raise ValueError(
                "When target_nodes is not None, it must have the same length"
                "as source_nodes"
            )

        super().__init__(**kwargs)
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        self.lengths = lengths
        self.include_smaller_paths = include_smaller_paths

    def value_defaults(self, data: torch_geometric.data.Data):
        """Sets default values for source_nodes and lengths if not provided."""
        if self.source_nodes is None:
            self.source_nodes = np.arange(data.num_nodes)
        if self.lengths is None:
            self.lengths = [2] * len(self.source_nodes)

    def find_hyperedges(self, data: torch_geometric.data.Data):
        """Finds hyperedges from paths between nodes in a graph."""
        G = torch_geometric.utils.convert.to_networkx(data, to_undirected=True)
        s_hyperedges = set()

        if self.target_nodes is None:  # all paths stemming from source nodes only
            for source, length in zip(self.source_nodes, self.lengths, strict=True):
                D, d_id2label, l_leafs = self.build_stemmingTree(G, source, length)
                s = self.extract_hyperedgesFromStemmingTree(D, d_id2label, l_leafs)
                s_hyperedges = s_hyperedges.union(s)

        else:  # paths from source_nodes to target_nodes or from source nodes only
            for source, target, length in zip(
                self.source_nodes, self.target_nodes, self.lengths, strict=True
            ):
                if target is None:
                    D, d_id2label, l_leafs = self.build_stemmingTree(G, source, length)
                    s = self.extract_hyperedgesFromStemmingTree(D, d_id2label, l_leafs)
                    s_hyperedges = s_hyperedges.union(s)
                else:
                    paths = list(
                        nx.all_simple_paths(
                            G, source=source, target=target, cutoff=length
                        )
                    )
                    if not self.include_smaller_paths:
                        paths = [path for path in paths if len(path) - 1 == length]
                    s_hyperedges = s_hyperedges.union({frozenset(x) for x in paths})
        return s_hyperedges

    def lift_topology(self, data: torch_geometric.data.Data):
        if self.source_nodes is None or self.lengths is None:
            self.value_defaults(data)
        s_hyperedges = self.find_hyperedges(data)
        indices = [[], []]
        for edge_id, x in enumerate(s_hyperedges):
            indices[1].extend([edge_id] * len(x))
            indices[0].extend(list(x))
        incidence = torch.sparse_coo_tensor(
            indices, torch.ones(len(indices[0])), (len(data.x), len(s_hyperedges))
        )
        return {
            "incidence_hyperedges": incidence,
            "num_hyperedges": len(s_hyperedges),
            "x_0": data.x,
        }

    def build_stemmingTree(self, G, source_root, length, verbose=False):
        """Creates a directed tree from a source node with paths of a given length."""
        d_id2label = {}
        stack = []
        D = nx.DiGraph()
        n_id = 0
        D.add_node(n_id)
        d_id2label[n_id] = source_root
        stack.append(n_id)
        n_id += 1
        l_leafs = []
        while len(stack) > 0:
            node = stack.pop()
            neighbors = list(G.neighbors(d_id2label[node]))
            visited_id = nx.shortest_path(D, source=0, target=node)
            visited_labels = [d_id2label[i] for i in visited_id]
            for neighbor in neighbors:
                if neighbor not in visited_labels:
                    D.add_node(n_id)
                    d_id2label[n_id] = neighbor
                    if len(visited_labels) < length:
                        stack.append(n_id)
                    elif len(visited_labels) == length:
                        l_leafs.append(n_id)
                    else:
                        raise ValueError("Visited labels length is greater than length")
                    D.add_edge(node, n_id)
                    n_id += 1
            if verbose:
                print("\nLoop Variables Summary:")
                print("nodes:", node)
                print("neighbors:", neighbors)
                print("visited_id:", visited_id)
                print("visited_labels:", visited_labels)
                print("stack:", stack)
                print("id2label:", d_id2label)
        return D, d_id2label, l_leafs

    def extract_hyperedgesFromStemmingTree(self, D, d_id2label, l_leafs):
        """From the root of the directed tree D,
        extract hyperedges from the paths to the leafs."""
        a_paths = np.array(
            [list(map(d_id2label.get, nx.shortest_path(D, 0, x))) for x in l_leafs]
        )
        s_hyperedges = {
            (frozenset(x)) for x in a_paths
        }  # set bc != paths can be same hpedge
        if self.include_smaller_paths:
            for i in range(a_paths.shape[1] - 1, 1, -1):
                a_paths = np.unique(a_paths[:, :i], axis=0)
                s_hyperedges = s_hyperedges.union({(frozenset(x)) for x in a_paths})
        return s_hyperedges
