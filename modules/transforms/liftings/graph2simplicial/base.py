import networkx as nx
import torch
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.lifting import GraphLifting
from modules.transforms.data_manipulations.manipulations import compute_invariance_r_minus_1_to_r, compute_invariance_r_to_r


class Graph2InvariantSimplicialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to simplicial complexes and including 
    invariances

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "graph2simplicial"
        self.signed = kwargs.get("signed", False)

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex, graph: nx.Graph
    ) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            simplicial_complex, self.complex_dim, signed=self.signed
        )

        feat_ind = {}
        adj_dict = {}
        inc_dict = {}


        # Dictionaries for each dimension
        for r in range(0, simplicial_complex.dim+1):
            feat_ind[r] = torch.tensor(list(simplicial_complex.incidence_matrix(r, index=True)[1].keys()), dtype=torch.int)
            adj_dict[r] = lifted_topology[f'adjacency_{r}']

        # Remove incidence relations where the symmetric difference between (r-1)-cell and r-cell > 1
        for r in range(1, simplicial_complex.dim+1):
            r_inc = lifted_topology[f'incidence_{r}'].to_dense()
            for i in range(0, feat_ind[r-1].size(0)):
                for j in range(0, feat_ind[r].size(0)):
                    set_1 = set(feat_ind[r-1][i].tolist())
                    set_2 = set(feat_ind[r][j].tolist())

                    sym_dif = set(set_1).symmetric_difference(set(set_2))
                    if len(sym_dif) > 1:
                        r_inc[i][j] = 0
            lifted_topology[f'incidence_{r}'] = r_inc.to_sparse()

        # Assign r-cell the incidences of (r+1)-cell so that we can compute invariances
        for r in range(0, simplicial_complex.dim):
            inc_dict[r] = lifted_topology[f'incidence_{r+1}']

            #inc_dict[r] = lifted_topology[f'incidence_{r}'] 

        inv_same_dict = compute_invariance_r_to_r(feat_ind, graph.pos, adj_dict)
        inv_low_high_dict = compute_invariance_r_minus_1_to_r(feat_ind, graph.pos, inc_dict)

        # Set invariances in data
        # Fix for the mismatch in computing invariances above
        for r in range(0, simplicial_complex.dim+1):
            if r != simplicial_complex.dim:
                lifted_topology[f'inv_same_{r}'] = inv_same_dict[r]
            if r > 0:
                lifted_topology[f'inv_low_high_{r}'] = inv_low_high_dict[r-1]

        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )
        # If new edges have been added during the lifting process, we discard the edge attributes
        if self.contains_edge_attr and simplicial_complex.shape[1] == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                list(simplicial_complex.get_simplex_attributes("features", 1).values())
            )
        return lifted_topology



class Graph2SimplicialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "graph2simplicial"
        self.signed = kwargs.get("signed", False)

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex, graph: nx.Graph
    ) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            simplicial_complex, self.complex_dim, signed=self.signed
        )
        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )
        # If new edges have been added during the lifting process, we discard the edge attributes
        if self.contains_edge_attr and simplicial_complex.shape[1] == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                list(simplicial_complex.get_simplex_attributes("features", 1).values())
            )
        return lifted_topology
