import torch
from toponetx import CombinatorialComplex

from modules.data.utils.utils import get_combinatorial_complex_connectivity
from modules.transforms.liftings.lifting import HypergraphLifting


class Hypergraph2CombinatorialLifting(HypergraphLifting):
    r"""Abstract class for lifting hypergraphs to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "hypergraph2combinatorial"

    def _get_lifted_topology(self, combinatorial_complex: CombinatorialComplex) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        combinatorial_complex : CombinatorialComplex
            The combinatorial complex.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_combinatorial_complex_connectivity(combinatorial_complex)

        # Feature liftings

        features = combinatorial_complex.get_cell_attributes("features")

        for i in range(combinatorial_complex.dim + 1):
            x = [
                feat
                for cell, feat in features
                if combinatorial_complex.cells.get_rank(cell) == i
            ]
            if x:
                lifted_topology[f"x_{i}"] = torch.stack(x)
            else:
                num_cells = len(combinatorial_complex.skeleton(i))
                lifted_topology[f"x_{i}"] = torch.zeros(num_cells, 1)

        return lifted_topology
