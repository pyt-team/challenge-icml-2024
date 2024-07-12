import torch
import numpy as np
import networkx as nx
import torch_geometric
import pyflagsercount as pfc
from toponetx.classes import CombinatorialComplex
from modules.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class SimplicialPathsLifting(Graph2CombinatorialLifting):
    def __init__(
        self, d1, d2, q, i, j, complex_dim=2, chunk_size=1024, threshold=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.d1 = d1
        self.d2 = d2
        self.q = q
        self.i = i
        self.j = j
        self.complex_dim = complex_dim
        self.chunk_size = chunk_size
        self.threshold = threshold

    def _get_complex_connectivity(
        self, combinatorial_complex, adjacencies, incidences, max_rank
    ):
        practical_shape = list(
            np.pad(
                list(combinatorial_complex.shape),
                (0, max_rank + 1 - len(combinatorial_complex.shape)),
            )
        )
        connectivity = {}
        connectivity["shape"] = practical_shape
        for adj in adjacencies:
            connectivity_info = "adjacency"
            if adj[0] < adj[1]:
                connectivity[f"{connectivity_info}_{adj[0]}_{adj[1]}"] = (
                    torch.from_numpy(
                        (
                            combinatorial_complex.adjacency_matrix(
                                adj[0], adj[1]
                            ).todense()
                        )
                    )
                    .to_sparse()
                    .float()
                )
            else:
                connectivity[f"{connectivity_info}_{adj[0]}_{adj[1]}"] = (
                    torch.from_numpy(
                        (
                            combinatorial_complex.coadjacency_matrix(
                                adj[0], adj[1]
                            ).todense()
                        )
                    )
                    .to_sparse()
                    .float()
                )
        for inc in incidences:
            connectivity_info = "incidence"
            connectivity[f"{connectivity_info}_{inc[0]}_{inc[1]}"] = (
                torch.from_numpy(
                    (combinatorial_complex.incidence_matrix(inc[0], inc[1]).todense())
                )
                .to_sparse()
                .float()
            )
        return connectivity

    def _get_lifted_topology(
        self, combinatorial_complex: CombinatorialComplex, graph: nx.Graph
    ) -> dict:
        r"""Returns the lifted topology.
        Parameters
        ----------
        cell_complex : CellComplex
            The cell complex.
        graph : nx.Graph
            The input graph.
        Returns
        -------
        dict
            The lifted topology.
        """
        adjacencies = [[0, 1]]
        incidences = [[0, 2]]
        lifted_topology = self._get_complex_connectivity(
            combinatorial_complex, adjacencies, incidences, self.complex_dim
        )

        feat = torch.stack(list(nx.get_node_attributes(graph, "features").values()))
        lifted_topology["x_0"] = feat
        lifted_topology["x_2"] = torch.matmul(
            lifted_topology["incidence_0_2"].t(), feat
        )

        return lifted_topology

    def _create_flag_complex_from_dataset(self, dataset, complex_dim=2):
        dataset_digraph = nx.DiGraph()

        dataset_digraph.add_edges_from(
            list(zip(dataset.edge_index[0].tolist(), dataset.edge_index[1].tolist(), strict=False))
        )

        dfc = DirectedQConnectivity(dataset_digraph, complex_dim)

        return dfc

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:

        FlG = self._create_flag_complex_from_dataset(data, complex_dim=2)

        indices = FlG.qij_adj(
            FlG.complex[self.d1],
            FlG.complex[self.d2],
            self.q,
            self.i,
            self.j,
            self.chunk_size,
        )

        G = self._generate_graph_from_data(data)
        paths = FlG.find_paths(indices, self.threshold)

        cc = CombinatorialComplex(G)

        for p in paths:  # retrieve nodes that compose each path
            cell = list()
            for c in p:
                cell += list(FlG.complex[2][c].numpy())
            cell = list(set(cell))  # remove duplicates

            cc.add_cell(cell, rank=2)

        return self._get_lifted_topology(cc, G)


class DirectedQConnectivity:
    r"""Let :math:`G=(V,E)` be a directed graph. The directed flag complex of
    :math:`G` :math:`dFl(G)` is the ordered simplicial complex whose
    :math:`k`-simplices vertices are all totally ordered :math:`(k+1)`-cliques,
    i.e. :math:`(v_0, \dots, v_n)` such that :math:`(v_i, v_j) \in E` for
    all :math:`i \leq j`. This class provides a way to compute the directed
    flag complex of a directed graph, compute the qij-connectivity of
    the complex and find the maximal simplicial paths arising from the
    qij-connectivity.

    Parameters
    ----------
    digraph : nx.DiGraph
        The directed graph to compute the directed flag complex of.
    complex_dim : int
        The maximum dimension of the complex to compute.
    flagser_num_threads : int, optional
        The number of threads to use in the flagser computation. Default is 4.

    References
    ----------

    .. [1] Henri Riihïmaki. Simplicial q-Connectivity of Directed Graphs
    with Applications to Network Analysis. doi:10.1137/22M1480021.

    .. [2] D. Lütgehetmann, D. Govc, J.P. Smith, and R. Levi. Computing
    persistent homology of directed flag complexes. arXiv:1906.10458.
    """

    complex: dict[int, set[tuple]]

    def __init__(
        self, digraph: nx.DiGraph, complex_dim: int = 2, flagser_num_threads: int = 4
    ):

        self.digraph = digraph
        self.complex_dim = complex_dim

        sparse_adjacency_matrix = nx.to_scipy_sparse_array(digraph, format="csr")

        self.X = pfc.flagser_count(
            sparse_adjacency_matrix,
            threads=flagser_num_threads,
            return_simplices=True,
            max_dim=self.complex_dim,
            compressed=False,
        )

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # else torch.device("cpu") #Server

        # self.device = torch.device("mps")  # my macbook
        self.complex = self.X["simplices"]

        self.complex[0] = torch.tensor(
            [[node] for node in digraph.nodes], device=self.device
        )
        self.complex[1] = torch.tensor(
            [list(edge) for edge in digraph.edges], device=self.device
        )
        self.complex = [
            torch.tensor(item, device=self.device) if i >= 2 else item
            for i, item in enumerate(self.complex)
        ]

    def _d_i_batched(self, i: int, simplices: torch.tensor) -> torch.tensor:
        r"""Compute the face map :math:`d_i` of the simplices in the batched
        simplices tensor. The map :math:`d_i` removes a vertex at position
        :math:`min\{i, dim(\sigma)\}` for each simplex :math:`\sigma` in the
        batch.

        Parameters
        ----------
        i : int
            The index of the face map.
        simplices : torch.Tensor, shape=(batch_size, n_vertices)
            The batch of simplices.

        Returns
        -------
        d_i : torch.Tensor, shape=(batch_size, n_vertices-1)
            The batch of simplices after applying the face map :math:`d_i`.

        References
        ----------
        .. [1] Henri Riihïmaki. Simplicial q-Connectivity of Directed
        Graphs with Applications to Network Analysis. doi:10.1137/22M1480021.
        """

        batch_size, n_vertices = simplices.shape
        indices = torch.arange(
            n_vertices, device=simplices.device
        )  # Allocated on the same device as `simplices`
        # Create a mask that excludes the i-th vertex
        mask = indices != min(i, n_vertices - 1)
        # Use advanced indexing to select vertices while preserving the
        # batch structure
        d_i = simplices[:, mask]
        return d_i

    def _gen_q_faces_batched(self, simplices: torch.tensor, c: int) -> torch.tensor:
        r"""Compute the :math:`q`-dimensional faces of the simplices in the
        batched simplices tensor, where :math:c represents the cardinality
        of the faces to compute and :math:q=c-1.

        Parameters
        ----------
        simplices : torch.Tensor, shape=(batch_size, n_vertices)
            The simplices tensor.
        c : int
            The cardinality of the faces to compute.

        Returns
        -------
        faces : torch.Tensor, shape=(batch_size, n_faces, c)
            The :math:q-dimensional faces of the simplices tensor.
        """

        combinations = torch.combinations(
            torch.tensor(range(simplices.size(1)), device=self.device), c
        )

        return simplices[:, combinations]

    def _multiple_contained_chunked(
        self, sigmas: torch.Tensor, taus: torch.Tensor, chunk_size: int = 1024
    ) -> torch.Tensor:
        r"""Compute the adjacency matrix induced by the relation
        :math:`\sigma_i \subseteq \tau_j`. This function is chunked to avoid
        memory issues.

        Parameters
        ----------
        sigmas : torch.Tensor, shape=(Ns, cs)
            The first simplices tensor.
        taus : torch.Tensor, shape=(Nt, ct)
            The second simplices tensor.
        chunk_size : int, optional
            The size of the chunks to process. Default is 1024.

        Returns
        -------
        A : torch.sparse_coo_tensor, shape=(Ns, Nt)
            Adjacency matrix such that :math:`A(i,j) = 1` if
            :math:`\sigma_i \subseteq \tau_j:`, and :math:`0` otherwise.
        """

        Ns, cs = sigmas.size()
        Nt, ct = taus.size()

        # If cs > ct, no sigma can be contained in any tau.
        if cs > ct:
            return torch.sparse_coo_tensor(
                torch.empty([2, 0], dtype=torch.long),
                [],
                size=(Ns, Nt),
                dtype=torch.bool,
            )

        # Generate faces of taus
        faces = self._gen_q_faces_batched(taus, cs)
        Nf = faces.size(1)
        total_faces = Nt * Nf

        indices = []

        # Process in chunks for memory efficiency purposes.
        for i in range(0, Ns, chunk_size):
            end_i = min(i + chunk_size, Ns)
            sigmas_chunk = sigmas[i:end_i]  # Shape: [min(chunk_size,
            # remaining Ns), cs]

            temp_true_indices = []

            # Compute diffs and matches for this chunk
            for j in range(0, total_faces, chunk_size):
                end_j = min(j + chunk_size, total_faces)
                faces_chunk = faces.view(-1, cs)[
                    j:end_j
                ]  # Shape: [min(chunk_size, remaining faces), cs]

                # Broadcasting happens here with much smaller tensors
                diffs = sigmas_chunk.unsqueeze(1) - faces_chunk.unsqueeze(
                    0
                )  # shape: [min(chunk_size, remaining Ns), min(chunk_size,
                # remaining faces), cs]

                matches = (
                    diffs.abs().sum(dim=2) == 0
                )  # shape: [min(chunk_size, remaining Ns),  min(chunk_size,
                # remaining faces)]

                # (end_i - i) is the number of sigmas in the chunk.
                # (end_j - j) // Nf is the number of taus in the chunk
                # Nf is the number of faces in each tau of dimension equal
                # to the dimension of the simplices in sigma.
                matches_reshaped = matches.view(end_i - i, (end_j - j) // Nf, Nf)

                matches_aggregated = matches_reshaped.any(dim=2)

                # Update temporary result for this chunk of sigmas
                if matches_aggregated.nonzero(as_tuple=False).size(0) > 0:
                    temp_indices = matches_aggregated.nonzero(as_tuple=False).T
                    temp_indices[0] += i  # Adjust sigma indices for chunk
                    # offset
                    temp_indices[1] += j // Nf  # Adjust tau indices for
                    # chunk offset
                    temp_true_indices.append(temp_indices)

            if temp_true_indices:
                indices.append(torch.cat(temp_true_indices, dim=1))

        if indices:
            indices = torch.cat(indices, dim=1)
        else:
            indices = torch.empty([2, 0], dtype=torch.long)

        A = torch.sparse_coo_tensor(
            indices,
            torch.ones(indices.size(1), dtype=torch.bool),
            size=(Ns, Nt),
            device="cpu",
        )

        return A

    def _alpha_q_contained_sparse(
        self, sigmas: torch.Tensor, taus: torch.Tensor, q: int, chunk_size: int = 1024
    ) -> torch.Tensor:
        r"""Compute the adjacency matrix induced by the relation
        :math:`\sigma_i \sim \tau_j \Leftrightarrow \exists \alpha_q
        \subseteq \sigma_i \cap \tau_j`. This function is chunked to avoid
        memory issues.

        Parameters
        ----------
        sigmas : torch.Tensor, shape=(Ns, cs)
            The first simplices tensor.
        taus : torch.Tensor, shape=(Nt, ct)
            The second simplices tensor.
        q : int
            The dimension of the alpha_q simplices.
        chunk_size : int, optional
            The size of the chunks to process. Default is 1024.

        Returns
        -------
        A : torch.sparse_coo_tensor, shape=(Ns, Nt)
            Adjacency matrix. :math:`A(i,j) = 1` if
            there exists :math:`\alpha_q \in \Sigma_q` such that
            :math:`\alpha_q \subseteq \sigma_i \cap \tau_j:`, and :math:`0`
            otherwise.
        """

        alpha_q_in_sigmas = self._multiple_contained_chunked(
            self.complex[q], sigmas, chunk_size
        ).to(torch.float)

        alpha_q_in_taus = self._multiple_contained_chunked(
            self.complex[q], taus, chunk_size
        ).to(torch.float)

        # Compute the intersection of the two sparse tensors to get the
        # alpha_q contained in both sigmas and taus

        intersect = torch.sparse.mm(alpha_q_in_sigmas.t(), alpha_q_in_taus)

        values = torch.ones(intersect._indices().size(1))

        A = torch.sparse_coo_tensor(
            intersect._indices(),
            values,
            dtype=torch.bool,
            size=(sigmas.size(0), taus.size(0)),
        )

        return A

    def qij_adj(
        self,
        sigmas: torch.tensor,
        taus: torch.tensor,
        q: int,
        i: int,
        j: int,
        chunk_size: int = 1024,
    ):
        r"""Compute the adjacency matrix associated with the :math:`(q,
        d_i, d_j)`-connectivity relation of two not necessarily distinct
        pairs of skeletons of the complex.

        Parameters
        ----------
        sigmas : torch.Tensor, shape=(Ns, cs)
           The first batch of simplices corresponds to a skeleton of the
           complex.
        taus : torch.Tensor, shape=(Nt, ct)
            The second batch of simplices corresponds to a skeleton of the
            complex.
        q : int
            First parameter of the qij-connectivity relation.
        i : int
            Second parameter of the qij-connectivity relation. Determines
            the first face map of the ordered pair of face maps.
        j : int
            Third parameter of the qij-connectivity relation. Determines the
            second face map of the ordered pair of face maps.
        offset : torch.Tensor, shape=(2, 1)
            The initial indices to add to the computed indices.
        chunk_size : int, optional
            The size of the chunks to process. Default is 1024.

        Returns
        -------
        indices : torch.Tensor, shape=(2, N)
            The indices of the qij-connected simplices of the pair of
            skeletons.
        """

        if q > self.complex_dim:
            raise ValueError("q has to be lower than the complex dimension")

        di_sigmas = self._d_i_batched(i, sigmas)
        dj_taus = self._d_i_batched(j, taus)

        contained = self._multiple_contained_chunked(sigmas, taus, chunk_size)

        alpha_q_contained = self._alpha_q_contained_sparse(
            di_sigmas, dj_taus, q, chunk_size
        )

        indices = (
            torch.cat(
                (contained._indices().t(), alpha_q_contained._indices().t()), dim=0
            )
            .unique(dim=0)
            .t()
        )

        return indices

    def find_paths(self, indices: torch.tensor, threshold: int):
        r"""Find the paths in the adjacency matrix associated with the
        :math:`(q,
        d_i, d_j)`-connectivity relation with length longer than a threshold.
        Parameters
        ----------
        indices : torch.Tensor, shape=(2, N)
           The indices of the qij-connected simplices of the pair of skeletons.
        threshold : int
            The length threshold to select paths


        Returns
        -------
        paths : List[List]
            List of selected paths.
        """

        def dfs(node, adj_list, all_paths, path):

            if node not in adj_list:  # end of recursion
                if len(path) > threshold:
                    all_paths = add_path(path.copy(), all_paths)
                return

            only_loops = True
            for new_node in adj_list[node]:
                if new_node not in path:  # avoid cycles
                    only_loops = False
                    path.append(new_node)
                    dfs(new_node, adj_list, all_paths, path)
                    path.pop()

            if only_loops:  # then we have another longest path
                if len(path) > threshold:
                    all_paths = add_path(path.copy(), all_paths)

            return

        def edge_index_to_adj_list(edge_index):
            adj_list = {}
            for e in edge_index.T:
                if e[0].item() not in adj_list:
                    adj_list[e[0].item()] = [e[1].item()]
                else:
                    adj_list[e[0].item()].append(e[1].item())

            return adj_list

        def is_subpath(p1, p2):
            if len(p1) > len(p2):
                return False
            if len(p1) == len(p2):
                return p1 == p2
            else:
                diff = len(p2) - len(p1)
                for i in range(diff + 1):
                    if p2[i:i + len(p1)] == p1:
                        return True
            return False

        def add_path(new_path, all_paths):
            for path in all_paths:
                if is_subpath(new_path, path):
                    # don't add a subpath
                    return new_path
            # Check if some paths need to be removed
            for path in all_paths:
                if is_subpath(path, new_path):
                    all_paths.remove(path)
            all_paths.append(new_path)
            return all_paths

        adj_list = edge_index_to_adj_list(indices)

        all_paths = []

        for src in adj_list:
            path = [src]
            dfs(src, adj_list, all_paths, path)

        return all_paths
