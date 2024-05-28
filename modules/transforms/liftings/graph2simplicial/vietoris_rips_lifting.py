from itertools import combinations

import networkx as nx
import torch_geometric
import gudhi
from gudhi import SimplexTree
from toponetx.classes import SimplicialComplex
import torch


from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting
from modules.data.utils.utils import compute_invariance_r_minus_1_to_r, compute_invariance_r_to_r, SimplexData, get_complex_connectivity


def rips_lift(graph: torch_geometric.data.Data, dim: int, dis: float, fc_nodes: bool = True) -> SimplicialComplex:
    # create simplicial complex
    # Extract the node tensor and position tensor
    x_0, pos = graph.x, graph.pos

    # Create a list of each node tensor position 
    points = [pos[i].tolist() for i in range(pos.shape[0])]

    # Lift the graph to a Rips complex
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=dis)
    simplex_tree: SimplexTree  = rips_complex.create_simplex_tree(max_dimension=dim)

    # Add fully connected nodes to the simplex tree
    # (additionally connection between each pair of nodes u, v)

    if fc_nodes:
        nodes = [i for i in range(x_0.shape[0])]
        for edge in combinations(nodes, 2):
            simplex_tree.insert(edge)

    return SimplicialComplex.from_gudhi(simplex_tree)



class BaseSimplicialVRLifting(Graph2SimplicialLifting):
    r""" Base class for lifting graphs to simplicial complexes using Vietoris-Rips complex.
    """
    def __init__(self, delta: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex by using the Vietoris-Rips lift
        and assigns the features of the 0-rank simplices.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """


        # Lift graph to Simplicial Complex
        simplicial_complex = rips_lift(data, self.complex_dim, self.delta)

        # Retrieve features as a directory
        feature_dict = {}
        for i, node in enumerate(data.x):
            feature_dict[i] = node

        # Encode features in the simplex
        simplicial_complex.set_simplex_attributes(feature_dict, name='features')

        return self._get_lifted_topology(simplicial_complex, data)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        r"""Applies the full lifting (topology + features) to the input data.
        and incorporates the SimplexData structure for mini-batching.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        return SimplexData(**initial_data, **lifted_topology)

class InvariantSimplicialVietorisRipsLifting(BaseSimplicialVRLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the cliques as k-simplices.

    Parameters
    ----------
    distance : int, optional
        The distance for the Vietoris-Rips complex. Default is 0.
    **kwargs : optional
        Additional arguments for the class.
    """

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

        # Create feature dictionary
        for r in range(0, simplicial_complex.dim+1):
            lifted_topology[f'x_idx_{r}'] = feat_ind[r] = torch.tensor(simplicial_complex.skeleton(r), dtype=torch.int)
            lifted_topology[f'adjacency_{r}'] = adj_dict[r] = lifted_topology[f'adjacency_{r}'].to_dense().nonzero().t().contiguous()
            lifted_topology[f'incidence_{r}'] = lifted_topology[f'incidence_{r}'].to_dense().nonzero().t().contiguous()

        # Assign r-cell the incidences of (r+1)-cell so that we can compute invariances
        for r in range(0, simplicial_complex.dim):
            inc_dict[r] = lifted_topology[f'incidence_{r+1}']

        inv_same_dict = compute_invariance_r_to_r(feat_ind, graph.pos, adj_dict)
        inv_low_high_dict = compute_invariance_r_minus_1_to_r(feat_ind, graph.pos, inc_dict)

        # Set invariances in data
        # Fix for the mismatch in computing invariances above
        for r in range(0, simplicial_complex.dim+1):
            if r != simplicial_complex.dim:
                lifted_topology[f'inv_same_{r}'] = inv_same_dict[r]
            if r > 0:
                lifted_topology[f'inv_low_high_{r}'] = inv_low_high_dict[r-1]

        # Set features of the 0-cell
        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )
        return lifted_topology



class SimplicialVietorisRipsLifting(BaseSimplicialVRLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the cliques as k-simplices.

    Parameters
    ----------
    distance : int, optional
        The distance for the Vietoris-Rips complex. Default is 0.
    **kwargs : optional
        Additional arguments for the class.
    """

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

        # Transform matrices to edge_index format
        for r in range(0, simplicial_complex.dim+1):
            lifted_topology[f'adjacency_{r}'] = lifted_topology[f'adjacency_{r}'].to_dense().nonzero().t().contiguous()
            lifted_topology[f'incidence_{r}'] = lifted_topology[f'incidence_{r}'].to_dense().nonzero().t().contiguous()


        # Create feature dictionary
        for r in range(0, simplicial_complex.dim+1):
            lifted_topology[f'x_idx_{r}'] = torch.tensor(simplicial_complex.skeleton(r), dtype=torch.int)

        # Set features of the 0-cell
        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )

        return lifted_topology