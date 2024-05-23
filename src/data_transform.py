from typing import Dict
import torch
from torch_geometric.data import Data
import gudhi
from gudhi import SimplexTree

from modules.transforms.liftings.graph2simplicial.vietoris_rips_lift import SimplicialVietorisRipsLifting
from modules.transforms.liftings.graph2simplicial.alpha_complex_lift import SimplicialAlphaComplexLifting
from toponetx.classes import SimplicialComplex

qm9_to_ev = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']

class AlphaPreTransform(SimplicialAlphaComplexLifting):
    def __init__(self, **kwargs):
        self.target_name = kwargs['target_name']
        super().__init__(**kwargs)

    def __call__(self, data: Data) -> Data:
        data = prepare_data(data, self.target_name, qm9_to_ev)
        data = super().__call__(data)
        return data
class VietorisRipsPreTransfrom(SimplicialVietorisRipsLifting):
    def __init__(self, dis: float = 0, **kwargs):
        self.target_name = kwargs['target_name']
        super().__init__(dis=dis, **kwargs)

    def __call__(self, data: Data) -> Data:
        data = prepare_data(data, self.target_name, qm9_to_ev)
        data = super().__call__(data)
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
    


def prepare_data(graph: Data, target_name: str, qm9_to_ev: Dict[str, float]) -> Data:
    index = targets.index(target_name)
    graph.y = graph.y[0, index]
    one_hot = graph.x[:, :5]  # only get one_hot for cormorant
    # change unit of targets
    if target_name in qm9_to_ev:
        graph.y *= qm9_to_ev[target_name]

    Z_max = 9
    Z = graph.x[:, 5]
    Z_tilde = (Z / Z_max).unsqueeze(1).repeat(1, 5)

    graph.x = torch.cat((one_hot, Z_tilde * one_hot, Z_tilde * Z_tilde * one_hot), dim=1)

    return graph