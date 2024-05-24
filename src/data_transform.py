from typing import Dict
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import gudhi
from gudhi import SimplexTree

from toponetx.classes import SimplicialComplex

qm9_to_ev = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']


class InputPreprocTransform(T.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, data: Data) -> Data:
        one_hot = data.x[:, :5]
        Z_max = 9
        Z = data.x[:, 5]
        Z_tilde = (Z / Z_max).unsqueeze(1).repeat(1, 5)
        data.x = torch.cat((one_hot, Z_tilde * one_hot, Z_tilde * Z_tilde * one_hot), dim=1)

        return data

class LabelPreprocTransform(T.BaseTransform):
    def __init__(self, **kwargs):
        self.target_name = kwargs['target_name']
        super().__init__()

    def __call__(self, data: Data) -> Data:
        index = targets.index(self.target_name)
        data.y = data.y[0, index]

        if self.target_name in qm9_to_ev:
            data.y *= qm9_to_ev[self.target_name]
        return data

def filter_not_enough_simplices_alpha(graph: Data):

    pos = graph.pos

    # Create a list of each node tensor position 
    points = [pos[i].tolist() for i in range(pos.shape[0])]

    # Lift the graph to an AlphaComplex
    alpha_complex = gudhi.AlphaComplex(points=points)
    simplex_tree: SimplexTree = alpha_complex.create_simplex_tree(default_filtration_value=True)
    simplex_tree.prune_above_dimension(2)
    simplicial_complex = SimplicialComplex.from_gudhi(simplex_tree)
    return simplicial_complex.maxdim > 1