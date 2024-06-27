import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem import Descriptors
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
        dict of list
            The functional groups with the atom ids.
            Example:
            {
                "C(=O)O": [[0, 1, 2], [2, 3, 9]],
                "C(=O)N": [[3, 4, 5], [5, 6, 7]],
                ...
            }

        """
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
            The atom attributes with the atom id.
            Example:
            {
                0: {"atomic_num": 6, "symbol": "C", "degree": 3, "formal_charge": 0, "hybridization": "SP
                    "is_aromatic": False, "mass": 12.01, "chirality": "CHI_UNSPECIFIED"},

                1: {"atomic_num": 6, "symbol": "C", "degree": 2, "formal_charge": 0, "hybridization": "SP
                    "is_aromatic": False, "mass": 12.01, "chirality": "CHI_UNSPECIFIED"},
                ...
            }
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
            The bond attributes with the bond id.
            Example:
            {
                (0, 1): {"bond_type": "SINGLE", "is_conjugated": False, "is_stereo": False},
                (1, 2): {"bond_type": "DOUBLE", "is_conjugated": False, "is_stereo": False},
                ...
            }
        """
        attr = {}
        for bond in mol.GetBonds():
            attr[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
                        "bond_type": str(bond.GetBondType()),
                        "is_conjugated": bond.GetIsConjugated(),
                        "is_stereo": bond.GetStereo()
                    }
        return attr

    def get_ring_attributes(
            self, mol: Chem.Mol, rings: list
        ) -> dict:
        r"""Returns the ring attributes for each molecule.

        • Ring Size: Number of atoms that form the ring.
        • Aromaticity: A binary feature telling whether the ring is aromatic.
        • Has Heteroatom: A binary feature telling whether the ring contains a heteroatom,
          which is an atom other than carbon, such as nitrogen, oxygen, or sulfur.
        • Saturation: A binary feature telling whether the ring is saturated, meaning all
          the bonds between carbon atoms are single bonds.
        • Hydrophobicity: Tendency of the functional group to repel water.
             Compute based in logp.
        • Electrophilicity: Tendency of the functional group to accept electrons during a chemical reaction.
             Compute based on the number of hydrogen bonds acceptors.
        • Nucleophilicity: Tendency of the functional group to donate electrons during a chemical reaction.
             High nucleophilicity means the functional group readily donates electrons, affecting its reactivity
             with electrophiles. Compute based in number of hydrogen bond donors.
        • Polarity: Distribution of electron density within the functional group. High polarity can significantly
             affect solubility, reactivity, and interactions with other molecules.
             Compute based on the tpsa.
        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        rings : list
            The rings of the molecule.
            Example: [[0, 1, 2], [1, 2, 3], ...]

        Returns
        -------
        dict
            The ring attributes.
            Example:
            {
                (0, 1, 2): {"ring_size": 3, "aromaticity": False, "has_heteroatom": False,
                            "saturation": True, "hydrophobicity": 0.0, "electrophilicity": 0,
                            "nucleophilicity": 0, "polarity": 0.0},
                (1, 2, 3): {"ring_size": 3, "aromaticity": False, "has_heteroatom": False,
                            "saturation": True, "hydrophobicity": 0.0, "electrophilicity": 0,
                            "nucleophilicity": 0, "polarity": 0.0},
                ...
            }
        """
        attr = {}
        for ring in rings:
            ring_size = len(ring)
            mol_ring = Chem.MolFromSmiles("".join([mol.GetAtomWithIdx(atom).GetSymbol() for atom in ring]))
            aromaticity = all([atom.GetIsAromatic() for atom in mol_ring.GetAtoms()])
            has_heteroatom = any([atom.GetAtomicNum() != 6 for atom in mol_ring.GetAtoms()])
            saturation = all([bond.GetBondType() == Chem.rdchem.BondType.SINGLE for bond in mol_ring.GetBonds()])

            hydrophobicity = Descriptors.MolLogP(mol_ring)
            electrophilicity = Descriptors.NumHAcceptors(mol_ring)
            nucleophilicity = Descriptors.NumHDonors(mol_ring)
            polarity = Descriptors.TPSA(mol_ring)

            attr[tuple(ring)] = {
                "ring_size": ring_size,
                "aromaticity": aromaticity,
                "has_heteroatom": has_heteroatom,
                "saturation": saturation,
                "hydrophobicity": hydrophobicity,
                "electrophilicity": electrophilicity,
                "nucleophilicity": nucleophilicity,
                "polarity": polarity
            }

        return attr


    def get_functional_groups_attributes(
            self, mol: Chem.Mol, cliques: dict
        ) -> dict:
        r"""Returns the functional groups attributes for each molecule.

        • Conjugation: A binary feature telling whether the functional group
            is part of a conjugated system, which involves alternating double and single bonds.
        • Hydrophobicity: Tendency of the functional group to repel water.
             Compute based in logp.
        • Electrophilicity: Tendency of the functional group to accept electrons during a chemical reaction.
             Compute based on the number of hydrogen bonds acceptors.
        • Nucleophilicity: Tendency of the functional group to donate electrons during a chemical reaction.
             High nucleophilicity means the functional group readily donates electrons, affecting its reactivity
             with electrophiles. Compute based in number of hydrogen bond donors.
        • Polarity: Distribution of electron density within the functional group. High polarity can significantly
             affect solubility, reactivity, and interactions with other molecules.
             Compute based on the tpsa.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        cliques : dict
            The functional groups of the molecule.
            Example:
            {
                "C(=O)O": [[0, 1, 2], [2, 3, 9]],
                "C(=O)N": [[3, 4, 5], [5, 6, 7]],
                ...
            }

        Returns
        -------
        dict
            The functional groups attributes.
            Example:
            {
                (0, 1, 2): {"functional_group": "C(=O)O", "num_atoms": 3, "conjugation": False,
                            "hydrophobicity": 0.0, "electrophilicity": 0, "nucleophilicity": 0, "polarity": 0.0},
                (1, 2, 3): {"functional_group": "C(=O)O", "num_atoms": 3, "conjugation": False,
                            "hydrophobicity": 0.0, "electrophilicity": 0, "nucleophilicity": 0, "polarity": 0.0},
                ...
            }
        """

        attr = {}
        for fg, groups in cliques.items():
            fg_mol = Chem.MolFromSmarts(fg)
            Chem.SanitizeMol(fg_mol)  # Ensure all properties are calculated
            conjugation = all([bond.GetIsConjugated() for bond in fg_mol.GetBonds()])
            hydrophobicity = Descriptors.MolLogP(fg_mol)
            electrophilicity = Descriptors.NumHAcceptors(fg_mol)
            nucleophilicity = Descriptors.NumHDonors(fg_mol)
            polarity = Descriptors.TPSA(fg_mol)

            for group in groups:
                attr[tuple(group)] = {
                    "functional_group": fg,
                    "num_atoms": len(group),
                    "conjugation": conjugation,
                    "hydrophobicity": hydrophobicity,
                    "electrophilicity": electrophilicity,
                    "nucleophilicity": nucleophilicity,
                    "polarity": polarity
                }
        return attr

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
        ccc.set_cell_attributes(atom_attr, rank=0)

        # Set bond attributes
        bond_attr = self.get_bond_attributes(mol)
        ccc.set_cell_attributes(bond_attr, rank=1)

        # add rings as 2-cells
        rings = self.get_rings(mol)
        for ring in rings:
            ccc.add_cell(ring, rank=self.complex_dim)

        ring_attr = self.get_ring_attributes(mol, rings)
        ccc.set_cell_attributes(ring_attr, rank=self.complex_dim)

        # Hypergraph stuff
        # add functional groups and edges as hyperedges (rank = 1)
        # edges are like this: [[0, 1], [1, 2], ...]
        # cliques are like this: {"C(=O)O": [[0, 1, 2], [2, 3, 9]], ...}
        # aim: create a matrix where each row is an atom and each column is a hyperedge (cliques + edges)
        edges = [[edge[0], edge[1]] for edge in G.edges]
        cliques = self.get_functional_groups(mol)
        hyperedges = edges + list(cliques.values())
        num_hyperedges = len(hyperedges)
        incidence_hyperedges = torch.zeros(data.num_nodes, num_hyperedges)
        # create incidence matrix for hyperedges
        for i, edge in enumerate(hyperedges):
            for atom in edge:
                incidence_hyperedges[atom, i] = 1

        # set functional groups attributes
        if cliques:
            fg_attr = self.get_functional_groups_attributes(mol, cliques)
            # rank = 2, if not an error will be raised as there are more than 2 elements in the clique
            ccc.set_cell_attributes(fg_attr, rank=2)

        # Create the lifted topology dict for the cell complex
        ccc_lifted_topology = Graph2CellLifting._get_lifted_topology(self, ccc, G)

        # add hyperedges to the lifted topology
        ccc_lifted_topology["num_hyperedges"] = num_hyperedges
        ccc_lifted_topology["x_0"] = data.x
        ccc_lifted_topology["incidence_hyperedges"] = torch.Tensor(
            incidence_hyperedges
        ).to_sparse_coo()

        return ccc_lifted_topology
