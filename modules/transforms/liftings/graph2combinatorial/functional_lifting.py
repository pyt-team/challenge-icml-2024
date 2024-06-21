import torch
import torch_geometric
from rdkit import Chem
from toponetx.classes import CellComplex

from modules.transforms.liftings.graph2cell.base import Graph2CellLifting
from modules.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class CombinatorialFunctionalLifting(Graph2CombinatorialLifting):
    r"""Lifts r-cell features to r+1-cells by molecular rings.
    This rings are obtained by chemical knowledge from rdkit.
    Moreover, the functional groups are added as hyperedges.
    Functional groups are defined by SMARTS patterns in the
    corresponding lifting config file.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.complex_dim = 2

        self.functional_groups = kwargs.get("functional_groups", [])

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

    #######################################################
    ################### RINGS #############################
    #######################################################
    def get_rings(
            self, mol: Chem.Mol
        ) -> torch.Tensor:
        r"""Returns the ring information for each molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        Returns
        -------
        torch.Tensor
            The ring information.
        """

        return [list(ring) for ring in Chem.GetSymmSSSR(mol)]

    #######################################################
    ############## FUNCTIONAL GROUPS ######################
    #######################################################

    def get_functional_groups(
        self, mol: Chem.Mol
    ) -> dict:
        r"""Returns the functional groups for each molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        Returns
        -------
        dict

        """

        # return {fg: [list(match) for match in mol.GetSubstructMatches(Chem.MolFromSmarts(fg))]
        #             for fg in self.functional_groups if Chem.MolFromSmarts(fg)}

        cliques = {}
        for fg in self.functional_groups:
            fg_mol = Chem.MolFromSmarts(fg)
            if fg_mol:
                matches = mol.GetSubstructMatches(fg_mol)
                if matches:
                    cliques[fg] = [list(match) for match in matches]

        return cliques

    #######################################################
    ################### ATTRIBUTES ########################
    #######################################################
    def get_atom_attributes(
            self, mol: Chem.Mol
        ) -> dict:
        r"""Returns the atom attributes for each molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        Returns
        -------
        dict
            The atom attributes.
        """
        attr = {}
        for atom in mol.GetAtoms():
            attr[atom.GetIdx()] = {
                    "atomic_num": atom.GetAtomicNum(),
                    "symbol": atom.GetSymbol(),
                    "degree": atom.GetDegree(),
                    "formal_charge": atom.GetFormalCharge(),
                    "hybridization": str(atom.GetHybridization()),
                    "is_aromatic": atom.GetIsAromatic(),
                    "mass": atom.GetMass(),
                    "chirality": atom.GetChiralTag()
                }
        return attr

    def get_bond_attributes(
            self, mol: Chem.Mol
        ) -> torch.Tensor:
        r"""Returns the bond attributes for each molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        Returns
        -------
        dict
            The bond attributes.
        """
        attr = {}
        for bond in mol.GetBonds():
            attr[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
                        "bond_type": str(bond.GetBondType()),
                        "is_conjugated": bond.GetIsConjugated(),
                        "is_stereo": bond.GetIsStereo()
                    }
        return attr

    def get_ring_attributes(
            self, mol: Chem.Mol
        ) -> dict:
        r"""Returns the ring attributes for each molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        Returns
        -------
        dict
            The ring attributes.
        """
        # not implemented yet
        return {}

    #######################################################
    ################### LIFT ##############################
    #######################################################
    def lift_topology(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Lifts the topology of a graph to a combinatorial complex.
        Following the steps:
        1. Convert the data to a molecule and remove the hydrogens.
        2. Generate the graph from the molecule. Take into account that this graph
            will not be the same as the starting one, since the hydrogens are removed.
        3. Generate the cell complex from the graph adding the rings as 2-cells.
        4. Add functional groups as hyperedges.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data."""

        # Combinaotrial complex is a combination of a cell complex and a hypergraph
        # First create a cell complex and then add hyperedges

        G = self._generate_graph_from_data(data)
        ccc = CellComplex(G)
        mol = self._generate_mol_from_data(data)

        # Set atom attributes
        atom_attr = self.get_atom_attributes(mol)
        ccc.set_cell_attributes(atom_attr)

        # Set bond attributes
        bond_attr = self.get_bond_attributes(mol)
        ccc.set_cell_attributes(bond_attr)

        # add rings as 2-cells
        rings = self.get_rings(mol)
        for ring in rings:
            ccc.add_cell(ring, rank=self.complex_dim)

        ring_attr = self.get_ring_attributes(mol)
        ccc.set_cell_attributes(ring_attr)

        # Hypergraph stuff
        # add functional groups as hyperedges (rank = 1)
        cliques = self.get_functional_groups(mol)
        num_hyperedges = len(cliques)
        # create incidence matrix for hyperedges
        incidence_1 = torch.zeros(data.num_nodes, num_hyperedges)
        for i, group in enumerate(cliques.values()):
            for hyperedge in group:
                for atom in hyperedge:
                    incidence_1[atom, i] = 1

        # Create the lifted topology dict for the cell complex
        ccc_lifted_topology = Graph2CellLifting._get_lifted_topology(self, ccc, G)

        # add hyperedges to the lifted topology
        ccc_lifted_topology["num_hyperedges"] = num_hyperedges
        ccc_lifted_topology["x_0"] = data.x
        ccc_lifted_topology["incidence_hyperedges"] = torch.Tensor(
            incidence_1
        ).to_sparse_coo()

        return ccc_lifted_topology
