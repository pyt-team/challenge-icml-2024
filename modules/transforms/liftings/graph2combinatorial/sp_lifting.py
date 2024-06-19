import networkx as nx
import torch_geometric
import Graph2CombinatorialLifting

import networkx as nx
import pyflagsercount as pfc
import torch


class DirectedFlagComplex:
    r"""Let :math:`G=(V,E)` be a directed graph. The directed flag complex of
    :math:`G` :math:`dFl(G)` is the ordered simplicial complex whose
    :math:`k`-simplices vertices are all totally ordered :math:`(k+1)`-cliques,
    i.e. :math:`(v_0, \dots, v_n)` such that :math:`(v_i, v_j) \in E` for
    all :math:`i \leq j`. This class provides a way to compute the directed
    flag complex of a directed graph and to compute the qij-connectivity of
    the complex.

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

        # self.device = torch.device("cuda") if torch.cuda.is_available()
        # else torch.device("cpu") #Server

        self.device = torch.device("mps")  # my macbook
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

    def Bi_k(self, i, k):
        r"""Compute the boundary matrix :math:`B_{i,k}` of the complex. The
        boundary matrix is the matrix representation of the :math:`i`-th
        face map of the complex."""
        return self._multiple_contained_chunked(
            self.complex[k - 1], self._d_i_batched(i, self.complex[k])
        )

    def Ci_k(self, i, k):
        r"""Compute the coboundary matrix :math:`C_{i,k}` of the complex. The
        coboundary matrix is the matrix representation of the :math:`i`-th
        coboundary map of the complex."""

        if k + 1 > self.complex_dim:
            return []

        return self._d_i_batched(i, self.complex[k - 1])

    def Nij_k_down(self, i, j, k):
        N_k = self.complex[k].size(0)  # Number of k-simplices
        indices = self._block_qij(
            self.complex[k], self.complex[k], i, j, torch.tensor([[0], [0]])
        )
        return torch.SparseTensor(indices=indices, size=(N_k, N_k))

    def Nij_k_up(self, i, j, k):
        pass

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
        # Use advanced indexing to select vertices while preserving the batch structure
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
            Adjacency matrix such that :math:`A(i,j) = 1` if :math:`\sigma_i \subseteq
            \tau_j:`, and :math:`0` otherwise.
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
            sigmas_chunk = sigmas[i:end_i]  # Shape: [min(chunk_size, remaining Ns), ls]

            temp_true_indices = []

            # Compute diffs and matches for this chunk
            for j in range(0, total_faces, chunk_size * Nf):
                end_j = min(j + chunk_size * Nf, total_faces)
                faces_chunk = faces.view(-1, cs)[
                    j:end_j
                ]  # Shape: [min(chunk_size * Nt, remaining faces), ls]

                # Broadcasting happens here with much smaller tensors
                diffs = sigmas_chunk.unsqueeze(1) - faces_chunk.unsqueeze(
                    0
                )  # shape: [min(chunk_size, remaining Ns), # min(chunk_size * Nt, remaining faces), ls]

                matches = diffs.abs().sum(dim=2) == 0  # shape: [min(
                # chunk_size, remaining Ns), min(chunk_size * Nt, remaining
                # faces)]

                # (endi - i) is the number of sigmas in the chunk.
                # (endj - j)// Nf is the number of taus in the chunk
                # Nf is the number of faces in each tau of dimension equal to
                # the dimension of the simplices in sigma.
                matches_reshaped = matches.view(end_i - i, (end_j - j) // Nf, Nf)

                matches_aggregated = matches_reshaped.any(dim=2)

                # Update temporary result for this chunk of sigmas
                temp_true_indices.append(matches_aggregated.nonzero().T)

                # Concatenate tensors in temp_true_indices before appending
                if len(temp_true_indices) > 0:
                    true_indices_concat = torch.cat(temp_true_indices, dim=1)
                    # Adjust indices for the current chunk offset
                    true_indices_concat[0] += i  # Adjust sigma indices for chunk offset
                    true_indices_concat[1] += j  # Adjust tau indices for chunk offset

                    indices.append(true_indices_concat)

        indices = torch.cat(indices, dim=1)

        A = torch.sparse_coo_tensor(
            indices,
            torch.ones(indices.size(1)),
            size=(Ns, Nt),
            dtype=torch.bool,
            device="cpu",
        )

        return A

    def _alpha_q_contained_sparse(
        self, sigmas: torch.Tensor, taus: torch.Tensor, q: int, chunk_size:
            int = 1024
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

    def _block_qij(
        self,
        sigmas: torch.tensor,
        taus: torch.tensor,
        q: int,
        i: int,
        j: int,
        offset: torch.tensor,
        chunk_size: int = 1024,
    ):
        r"""Compute the adjacency matrix associated with the :math:`(q,
        d_i, d_j)`-connectivity relation of two not necessarily distinct
        pairs of skeletons of the complex.

        Parameters
        ----------
        sigmas : torch.Tensor, shape=(Ns, cs)
           The first batch of simplices corresponds to a skeleton of the complex.
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
            The indices of the qij-connected simplices of the pair of skeletons.
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

        return indices + offset

    def qij(self, q, i, j, chunk_size: int = 1024, path: str = None):
        r"""Compute the adjacency matrix associated to the :math:`(q,d_i,
        d_j)`-connectivity relation of the complex. Let :math:`(d_i,
        d_j)` be an ordered pair of face maps. Then, a pair of
        simplices :math:`(\sigma_i,\tau_j)` are :math:`(q,d_i,d_j)`-near if
        :math:`\sigma_i \subseteq \tau_j` or if there exists :math:`\alpha
        \in \Sigma_q` such that :math:`\alpha \subseteq \sigma_i` and
        :math:`\alpha \subseteq \tau_j`.

        Parameters
        ----------
        q : int
            First parameter of the qij-connectivity relation.
        i : int
            Second parameter of the qij-connectivity relation. Determines
            the first face map of the ordered pair of face maps.
        j : int
            Third parameter of the qij-connectivity relation. Determines the
            second face map of the ordered pair of face maps.
        chunk_size : int, optional
            The size of the chunks to process. Default is 1024.
        path : str, optional
            The path to save the indices of the adjacency matrix. Default is
            None.

        Returns
        -------
        preorder : torch.sparse_coo_tensor, shape=(N, N)
            The adjacency matrix associated to the :math:`(q, d_i,
            d_j)`-connectivity relation of the complex.
        """

        if q > self.complex_dim:
            raise ValueError("q has to be lower than the complex dimension")

        # TODO: Since the highest dimensional simplices of K are maximal,
        # they are dim(K)-connected only to themselves
        # TODO: Every q-simplex is connected to itself along any pair of face
        # maps.
        # TODO: If the ordered pair (\sigma,\tau) is qij-near, then, (\tau,
        # \sigma) is qji-near.
        # TODO: If (\sigma,\tau) is qii-near and (\tau,\sigma) is qii-near too.
        # TODO: A simplicial path is a sequence of simplices (simga_0,
        # sigma_1, ..., sigma_n) such that sigma_i and sigma_{i+1} are
        # qij-near.

        indices = []
        cells = self.complex[q:]
        dims = len(cells)
        ns = [len(skeleton) for skeleton in cells]
        N = sum(ns)

        for s in range(dims):
            for t in range(dims):
                sliding = torch.tensor([[sum(ns[:s])], [sum(ns[:t])]])

                indices.append(
                    self._block_qij(cells[s], cells[t], q, i, j, sliding, chunk_size)
                )

        indices = torch.cat(indices, dim=1).unique(dim=1)

        if path is not None:
            torch.save(indices, path)

        return indices


if __name__ == "__main__":

    G = nx.DiGraph()
    edges = [(i, i + offset) for i in range(100) for offset in (1, 2)]
    G.add_edges_from(edges)
    FlG = DirectedFlagComplex(G)

    edges = FlG.complex[1]
    triangles = FlG.complex[2]

    print(edges)
    print(FlG.Bi_k(0, 2))




class SPLifting(Graph2CombinatorialLifting):
    def __init__(self, **kwargs):
        pass 

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        pass


    

