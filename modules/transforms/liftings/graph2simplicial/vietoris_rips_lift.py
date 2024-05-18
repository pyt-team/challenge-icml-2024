from itertools import combinations

import networkx as nx
import torch_geometric
import gudhi
from gudhi import SimplexTree
from toponetx.classes import SimplicialComplex
import torch

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


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

class SimplicialVietorisRipsLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the cliques as k-simplices.

    Parameters
    ----------
    distance : int, optional
        The distance for the Vietoris-Rips complex. Default is 0.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, dis: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.dis = dis

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex by identifying the cliques as k-simplices.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)

        # Lift graph to Simplicial Complex
        simplicial_complex = rips_lift(graph, self.dim, self.dis)

        # Assign feature embeddings to the SimplicialComplex for 0-simplices (nodes)
        # and then for higher order n-simplices by taking the mean of the lower order simplices

        '''

        simplex_dict = {i: [] for i in range(self.dim+1)} 

        # Add the simplices for each n-dimension
        for simplex in simplicial_complex.simplices:
            dim = len(simplex) - 1
            simplex_dict[dim].append(torch.tensor(list(simplex)))
        simplex_dict = {k: torch.stack(v) for k, v in simplex_dict.items()} 

        # Cool dict comprehension to assign feature embeddings to each simplex
        simplex_feature_dict = {
            simplex: [] for simplex in simplicial_complex.simplices
        }


        for i in range(self.dim+1):
            z = []
            # For the k-component point in the i-simplices where k <= i
            for k in range(i+1):
                # Get the k-node indices of the i-simplices (i.e. the k-components of the i-simplices)
                z_i_idx = simplex_dict[i][:, k]
                # Get the node embeddings for the k-components of the i-simplices
                z_i = graph.x[z_i_idx]
                z.append(z_i)
            z = torch.stack(z, dim=2)
            # Mean along the simplex dimension
            z = z.mean(axis=2)
            # Tensor containing the components of the i-simplices
            n_simplices = simplex_dict[i]
            # Assign to each i-simplex the corresponding feature
            for l, simplex_tuple in enumerate(n_simplices):
                simplex_index = frozenset(simplex_tuple.tolist())
                simplex_feature_dict[simplex_index] = z[l]
        # Actually assign the embeddings
        simplicial_complex.set_simplex_attributes(simplex_feature_dict, name='feature')
        '''

        # TODO Add edge_attributes 

        return self._get_lifted_topology(simplicial_complex, graph)


if __name__ == '__main__':
    from torch_geometric.datasets import QM9

    dim = 2
    dis = 3


    data_root = f'./datasets/QM9_delta_{dis}_dim_{dim}'
    dataset = QM9(root=data_root)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw(g)

    
    graph = dataset[0]
    print(graph)
    #lift = VietorisRipsLifting(complex_dim=2, singed=False, dis=3)


