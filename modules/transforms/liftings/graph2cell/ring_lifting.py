import networkx as nx
import torch
import torch_geometric
from rdkit import Chem
from toponetx.classes import CellComplex

from modules.transforms.liftings.graph2cell.base import Graph2CellLifting


class CellRingLifting(Graph2CellLifting):
    r"""Lifts r-cell features to r+1-cells by molecular rings.
    This rings are obtained by chemical knowledge from rdkit.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.complex_dim = 2

    def get_rings(
            self, data: torch_geometric.data.Data | dict
        ) -> torch.Tensor:
        r"""Returns the ring information for each molecule.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch.Tensor
            The ring information.
        """

        # transform data to molecule and compute rings
        mol = self._generate_mol_from_data(data)
        rings = Chem.GetSymmSSSR(mol)

        return [list(ring) for ring in rings]

        
    def _generate_mol_from_data(
            self, data: torch_geometric.data.Data | dict
        ) -> Chem.Mol:
        r"""Converts the data to a molecule through the SMILES
            and removes the hydrogens.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        Chem.Mol
            The molecule.
        """
        mol = Chem.MolFromSmiles(data.smiles)
        return Chem.rdmolops.RemoveHs(mol)

    def lift_topology(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Lifts the topology of a graph to higher-order topological domains.
        Following the steps:
        1. Convert the data to a molecule and remove the hydrogens.
        2. Generate the graph from the molecule. Take into account that this graph
            will not be the same as the starting one, since the hydrogens are removed.
        3. Generate the cell complex from the graph.
        4. Add the rings as 2-cells.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data."""

        G = self._generate_graph_from_data(data)
        cell_complex = CellComplex(G)
        rings = self.get_rings(data)
        # add rings as 2-cells
        cell_complex.add_cells_from(
                rings, 
                rank=self.complex_dim
            )

        return self._get_lifted_topology(cell_complex, G)
