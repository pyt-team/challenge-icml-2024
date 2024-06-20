import networkx as nx
import torch
import torch_geometric
from rdkit import Chem
from toponetx.classes import CombinatorialComplex, CellComplex

from modules.transforms.liftings.graph2combinatorial.base import Graph2CombinatorialLifting
from modules.transforms.liftings.graph2cell.base import Graph2CellLifting

class CombinatorialRingCloseAtomsLifting(Graph2CombinatorialLifting):
    r"""Lifts r-cell features to r+1-cells by molecular rings.
    This rings are obtained by chemical knowledge from rdkit.
    Moreover, the close atoms are defined as the atoms that 
    are closer than a threshold distance. This atoms will be
    connected through an hyperedge.


    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.complex_dim = 2

        self.threshold_distance = kwargs.get("threshold_distance", 2)

    #######################################################
    ################### RINGS #############################
    #######################################################
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

    #######################################################
    ################ CLOSE ATOMS ##########################
    #######################################################
    def get_distance_matrix(
        self, data : torch_geometric.data.Data | dict
    ) -> torch.Tensor:
        r"""Computes the pairwise distances between atoms in a molecule.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data containing the positions of the atoms.

        Returns
        -------
        torch.Tensor
            The pairwise distance matrix.
        """
        # Get the positions of the atoms
        pos = data.pos

        # Compute the pairwise distances between the atoms
        distances = torch.cdist(pos, pos, p=2)

        return distances
    
    def find_close_atoms(
            self, data : torch_geometric.data.Data | dict
        ) -> list:
        r"""Finds the atoms that are close to each other within a molecule.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data containing the positions of the atoms.

        Returns
        -------
        list
            The list of close atoms.
        """
        # Get the distances between atoms including in transormed data
        distance_matrix = self.get_distance_matrix(data)

        # Get indices of atom pairs that are closer than the threshold
        close_atoms = []
        num_atoms = distance_matrix.size(0) # data.num_nodes
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if distance_matrix[i, j] < float(self.threshold_distance):
                    close_atoms.append((i, j))
        
        return close_atoms      #, distance_matrix
    
    def find_close_atom_groups(
        self, data : torch_geometric.data.Data | dict
    ) -> list:
        r"""Finds the groups of atoms that are close to each other within a molecule.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict

        Returns
        -------
        list
            The list of groups of close atoms.
        """
        # Get the indices of close atoms
        close_atoms = self.find_close_atoms(data)

        G = nx.Graph()
        G.add_edges_from(close_atoms)
        return [list(component) for component in nx.connected_components(G)]


    #######################################################
    ################### LIFT ##############################
    #######################################################
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

        # Combinaotrial complex is a combination of a cell complex and a hypergraph
        # First create a cell complex and then add hyperedges

        G = self._generate_graph_from_data(data)
        ccc = CellComplex(G)
        
        # add rings as 2-cells
        rings = self.get_rings(data)
        for ring in rings:
            ccc.add_cell(
                ring, 
                rank=self.complex_dim
            )
        
        # Hypergraph stuff
        # add close atoms as hyperedges (rank = 1)
        close_atoms = self.find_close_atom_groups(data)
        num_hyperedges = len(close_atoms)
        # create incidence matrix for hyperedges
        incidence_1 = torch.zeros(data.num_nodes, num_hyperedges)
        for i, group in enumerate(close_atoms):
            for node in group:
                incidence_1[node, i] = 1
        

        # Create the lifted topology dict for the cell complex
        ccc_lifted_topology = Graph2CellLifting._get_lifted_topology(self, ccc, G)

        # add hyperedges to the lifted topology
        ccc_lifted_topology["num_hyperedges"] = num_hyperedges
        ccc_lifted_topology["x_0"] = data.x
        ccc_lifted_topology["incidence_hyperedges"] = torch.Tensor(incidence_1).to_sparse_coo()

        return ccc_lifted_topology
