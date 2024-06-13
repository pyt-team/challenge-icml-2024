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

    def get_rings(self, mol: Chem.Mol) -> torch.Tensor:
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

        return [list(ring) for ring in Chem.GetSymmSSSR(mol)]

    def _generate_mol_from_data(self) -> Chem.Mol:
        r"""Converts the data to a molecule and removes the hydrogens.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        Chem.Mol
            The molecule.
        """
        # Not always the data will have the correct SMILE
        # Hence, the molecule will not be generated
        # These points should be removed from the dataset
        self.mol = Chem.MolFromSmiles(self.data.smiles)
        if self.mol is None:
            return None
        else:
            return Chem.rdmolops.RemoveHs(Chem.MolFromSmiles(self.data.smiles))

    def _generate_graph_from_mol(self, mol: Chem.Mol) -> nx.Graph:
        r"""Generates a NetworkX graph from the input molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The input molecule.

        Returns
        -------
        nx.Graph
            The graph.
        """
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), atom=atom.GetSymbol())

        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond=bond.GetBondTypeAsDouble(),
            )

        return G

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

        rings : torch.Tensor
            The ring information for each molecule

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data."""

        self.mol = self._generate_mol_from_data()
        if self.mol is None:
            pass  # remove that data point
        else:
            G = self._generate_graph_from_mol(self.mol)
            cell_complex = CellComplex(G)

            # add rings as 2-cells
            cell_complex.add_cells_from(self.get_rings(self.mol), rank=self.complex_dim)

            return self._get_lifted_topology(cell_complex, G)
