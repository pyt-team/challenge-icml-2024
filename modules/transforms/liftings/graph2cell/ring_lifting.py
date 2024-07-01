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

    def get_rings(self, data: torch_geometric.data.Data | dict) -> torch.Tensor:
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
        cell_complex.add_cells_from(rings, rank=self.complex_dim)

        # add attributes
        

        return self._get_lifted_topology(cell_complex, G)
