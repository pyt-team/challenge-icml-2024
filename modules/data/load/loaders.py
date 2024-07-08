import os
import random

import networkx as nx
import numpy as np
import requests
import rootutils
import torch
import torch_geometric
from Bio import PDB
from omegaconf import DictConfig

from modules.data.load.base import AbstractLoader
from modules.data.utils.concat2geometric_dataset import ConcatToGeometricDataset
from modules.data.utils.custom_dataset import CustomDataset
from modules.data.utils.utils import (
    load_cell_complex_dataset,
    load_hypergraph_pickle_dataset,
    load_manual_graph,
    load_simplicial_dataset,
)


class GraphLoader(AbstractLoader):
    r"""Loader for graph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    #######################################################################
    ############## Auxiliar functions for loading UniProt data ############
    #######################################################################

    def fetch_uniprot_ids(self) -> list[dict]:
        r"""Fetch UniProt IDs by its API under the parameters specified in the configuration file."""
        query_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": self.parameters.query,
            "format": self.parameters.format,
            "fields": self.parameters.fields,
            "size": self.parameters.size
        }

        response = requests.get(query_url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data from UniProt. Status code: {response.status_code}")
            return []

        data = response.text.strip().split("\n")[1:]
        proteins = [{"uniprot_id": row.split("\t")[0], "sequence_length": int(row.split("\t")[1])} for row in data]

        # Ensure we have at least the required proteins to sample from
        if len(proteins) >= self.parameters.size:
            sampled_proteins = random.sample(proteins, self.parameters.size)
        else:
            print(f"Only found {len(proteins)} proteins within the specified length range. Returning all available proteins.")
            sampled_proteins = proteins

        # save sampled proteins to a csv file
        # create directory if not exist
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.data_dir + "/uniprot_ids.csv", "w") as file:
            for protein in sampled_proteins:
                file.write(f"{protein}\n")

        return sampled_proteins

    def fetch_protein_mass(
            self, uniprot_id : str
        ) -> float:
        r"""Returns the mass of a protein given its UniProt ID.
        This will be used as our target variable.

        Parameters
        ----------
        uniprot_id : str
            The UniProt ID of the protein.

        Returns
        -------
        float
            The mass of the protein.
        """
        url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}"
        response = requests.get(url, headers={"Accept": "application/json"})
        if response.status_code == 200:
            data = response.json()
            return data.get("sequence", {}).get("mass")
        return None

    def fetch_alphafold_structure(
            self, uniprot_id : str
        ) -> str:
        r"""Fetches the AlphaFold structure for a given UniProt ID.
        Not all the proteins have a structure available.
        This ones will be descarded.

        Parameters
        ----------
        uniprot_id : str
            The UniProt ID of the protein.

        Returns
        -------
        str
            The path to the downloaded PDB file.
        """
        pdb_dir = self.data_dir + "/pdbs"
        os.makedirs(pdb_dir, exist_ok=True)
        file_path = os.path.join(pdb_dir, f"{uniprot_id}.pdb")

        if os.path.exists(file_path):
            print(f"PDB file for {uniprot_id} already exists.")
        else:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "w") as file:
                    file.write(response.text)
                print(f"PDB file for {uniprot_id} downloaded successfully.")
            else:
                print(f"Failed to fetch the structure for {uniprot_id}. Status code: {response.status_code}")
                return None
        return file_path

    def parse_pdb(
            self, file_path : str
        ) -> PDB.Structure:
        r"""Parse a PDB file and return a BioPython structure object.

        Parameters
        ----------
        file_path : str
            The path to the PDB file.

        Returns
        -------
        PDB.Structure
            The BioPython structure object.
        """

        return PDB.PDBParser(QUIET=True).get_structure("alphafold_structure", file_path)

    def residue_mapping(
            self, uniprot_ids : list[str]
        ) -> dict:
        r"""Create a mapping of residue types to unique integers.
        Each residue type will be represented as a one unique integer.
        There are 20 standard amino acids, so we will have 20 unique integers (at maximum).

        Parameters
        ----------
        uniprot_ids : list[str]
            The list of UniProt IDs to process.

        Returns
        -------
        dict
            The mapping of residue types to unique integers.
        """

        residue_map = {}
        residue_counter = 0

        # First pass: determine unique residue types
        for uniprot_id in uniprot_ids:
            pdb_file = self.fetch_alphafold_structure(uniprot_id)
            if pdb_file:
                structure = self.parse_pdb(pdb_file)
                residues = [residue for model in structure for chain in model for residue in chain]
                for residue in residues:
                    residue_type = residue.get_resname()
                    if residue_type not in residue_map:
                        residue_map[residue_type] = residue_counter
                        residue_counter += 1
        return residue_map

    def calculate_residue_ca_distances_and_vectors(
            self, structure : PDB.Structure
        ):
        r"""Calculate the distances between the alpha carbon atoms of the residues.
        Also, calculate the vectors between the alpha carbon and beta carbon atoms of each residue.

        Parameters
        ----------
        structure : PDB.Structure
            The BioPython structure object.

        Returns
        -------
        list
            The list of residues.
        dict
            The dictionary of alpha carbon coordinates.
        dict
            The dictionary of beta carbon vectors.
        np.ndarray
            The matrix of distances between the residues.
        """

        residues = [residue for model in structure for chain in model for residue in chain]
        ca_coordinates = {}
        cb_vectors = {}
        distances = np.zeros((len(residues), len(residues)))

        for i, residue in enumerate(residues):
            if "CA" in residue:
                ca_coord = residue["CA"].get_coord()
                residue_type = residue.get_resname()
                residue_number = residue.get_id()[1]
                key = f"{residue_type}_{residue_number}" # this id is unique inside each protein
                ca_coordinates[key] = ca_coord

                # Not all residues have a CB atom
                cb_vectors[key] = residue["CB"].get_coord() - ca_coord if "CB" in residue else None


            for j in range(i + 1, len(residues)):
                if "CA" in residues[j]:
                    ca_coord2 = residues[j]["CA"].get_coord()
                    dist = np.linalg.norm(ca_coord - ca_coord2)
                    distances[i, j] = dist
                    distances[j, i] = dist

        return residues, ca_coordinates, cb_vectors , distances

    def calculate_vector_angle(
            self, v1 : np.ndarray, v2 : np.ndarray
        ) -> float:
        r"""Calculate the angle between two vectors.

        Parameters
        ----------
        v1 : np.ndarray
            The first vector.
        v2 : np.ndarray
            The second vector.

        Returns
        -------
        float
            The angle between the two vectors.
        """

        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    def calculate_edges(
            self, ca_coordinates : dict, cb_vectors : dict, distances : np.ndarray
        ) -> list:
        r"""Calculate the edges between the residues based on the distances and angles between them.

        Parameters
        ----------
        ca_coordinates : dict
            The dictionary of alpha carbon coordinates.
        cb_vectors : dict
            The dictionary of beta carbon vectors.
        distances : np.ndarray
            The matrix of distances between the residues.

        Returns
        -------
        list
            The list of edges.
        """

        edges = set()  # Use a set to track unique edges
        keys = list(ca_coordinates.keys())  # Which represent different residues

        for i in range(len(keys) - 1):
            key1 = keys[i]
            for j in range(i + 1, len(keys)):
                key2 = keys[j]
                dist = distances[i, j]
                angle = self.calculate_vector_angle(cb_vectors[key1], cb_vectors[key2]) if cb_vectors[key1] is not None and cb_vectors[key2] is not None else None

                # If they are sequential, add the edge directly
                # If they are not sequential, add the edge if the distance is less than the threshold
                if j == i + 1 or (dist < self.parameters.threshold and angle and angle < 90):
                    edges.add((key1, key2, dist, angle))

        return sorted(edges, key=lambda x: (int(x[0].split("_")[1]), int(x[1].split("_")[1])))  # Sort edges based on the ID after the underscore


    def  create_torch_geometric_data(
            self, residues : list, ca_coordinates : dict, residue_map : dict, cb_vectors : dict, edges : list, y : float
        ) -> None:
        r"""Create a torch_geometric.data.Data object from the protein data.

        Parameters
        ----------
        residues : list
            The list of residues.
        ca_coordinates : dict
            The dictionary of alpha carbon coordinates.
        residue_map : dict
            The mapping of residue types to unique integers.
        cb_vectors : dict
            The dictionary of beta carbon vectors.
        edges : list
            The list of edges.
        y : float
            The target variable.

        Returns
        -------
        torch_geometric.data.Data
            The torch_geometric.data.Data object.
        """

        keys = list(ca_coordinates.keys()) # residue name
        pos = [ca_coordinates[key] for key in keys]
        # check if cb_vectors exists for the key
        # Not all nodes have carbon beta, hence not all of them will have an attribute
        node_attr = [cb_vectors[key] for key in keys if cb_vectors[key] is not None]

        # Make One-Hot encoding of residue types, keep it in node_key
        num_residues = len(residue_map)

        node_key = []
        one_hot = torch.zeros(num_residues)
        for residue in residues:
            residue_type = residue.get_resname()
            one_hot[residue_map[residue_type]] = 1
            node_key.append(one_hot)
        # Does the same as the for loop above:
        # node_key = [torch.zeros(num_residues).scatter_(0, torch.tensor([residue_map[residue.get_resname()]]), 1) for residue in residues]


        node_map = {key: i for i, key in enumerate(keys)}
        # Set the edges
        edge_index = [[node_map[edge[0]], node_map[edge[1]]] for edge in edges]
        # Adding distance and angle as edge attributes
        edge_attr = [[edge[2], edge[3]] if edge[3] is not None else [edge[2], 0] for edge in edges]

        for edge in edges:
            edge_index.append([node_map[edge[0]], node_map[edge[1]]])
            # edge_attr.append([edge[2], edge[3]])
            if edge[3] is not None:
                edge_attr.append([edge[2], edge[3]])
            else:
                edge_attr.append([edge[2], 500]) # since to vectors can never have 500 as an angle

        # Create a graph
        G = nx.Graph()

        # add vertices
        G.add_nodes_from(node_map.values())

        # add edges
        G.add_edges_from(edge_index)
        G.to_undirected()
        edge_list = torch.tensor(list(G.edges)).T.long()

        # Convert to torch tensors
        pos = torch.tensor(np.array(pos), dtype=torch.float)
        node_key = torch.stack(node_key)
        node_attr = torch.tensor(np.array(node_attr), dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor([y], dtype=torch.float)

        return torch_geometric.data.Data(
            x=node_key,
            pos=pos,
            node_attr=node_attr,
            edge_index=edge_list,
            edge_attr=edge_attr,
            y = y
        )

    def load(self) -> torch_geometric.data.Dataset:
        r"""Load graph dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        # Define the path to the data directory
        root_folder = rootutils.find_root()
        root_data_dir = os.path.join(root_folder, self.parameters["data_dir"])

        self.data_dir = os.path.join(root_data_dir, self.parameters["data_name"])
        if (
            self.parameters.data_name.lower() in ["cora", "citeseer", "pubmed"]
            and self.parameters.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=root_data_dir,
                name=self.parameters["data_name"],
            )

        elif self.parameters.data_name in [
            "MUTAG",
            "ENZYMES",
            "PROTEINS",
            "COLLAB",
            "IMDB-BINARY",
            "IMDB-MULTI",
            "REDDIT-BINARY",
            "NCI1",
            "NCI109",
        ]:
            dataset = torch_geometric.datasets.TUDataset(
                root=root_data_dir,
                name=self.parameters["data_name"],
                use_node_attr=False,
            )

        elif self.parameters.data_name in ["ZINC", "AQSOL"]:
            datasets = []
            for split in ["train", "val", "test"]:
                if self.parameters.data_name == "ZINC":
                    datasets.append(
                        torch_geometric.datasets.ZINC(
                            root=root_data_dir,
                            subset=True,
                            split=split,
                        )
                    )
                elif self.parameters.data_name == "AQSOL":
                    datasets.append(
                        torch_geometric.datasets.AQSOL(
                            root=root_data_dir,
                            split=split,
                        )
                    )
            # The splits are predefined
            # Extract and prepare split_idx
            split_idx = {"train": np.arange(len(datasets[0]))}
            split_idx["valid"] = np.arange(
                len(datasets[0]), len(datasets[0]) + len(datasets[1])
            )
            split_idx["test"] = np.arange(
                len(datasets[0]) + len(datasets[1]),
                len(datasets[0]) + len(datasets[1]) + len(datasets[2]),
            )
            # Join dataset to process it
            dataset = datasets[0] + datasets[1] + datasets[2]
            dataset = ConcatToGeometricDataset(dataset)

        elif self.parameters.data_name in ["UniProt"]:
            """
            The UniProt dataset is a custom dataset that is created by fetching data from the UniProt API.
            The dataset is created by fetching a list of proteins based on a query and then fetching the structure
            of each protein using the AlphaFold API. The dataset is then created by creating a graph for each protein
            where the nodes are the residues and the edges are the connected residues. The target variable is the mass
            of the protein.
            """
            datasets = []
            protein_data = self.fetch_uniprot_ids()
            uniprot_ids = [protein["uniprot_id"] for protein in protein_data]
            # Determine unique residue types and create a mapping
            residue_map = self.residue_mapping(uniprot_ids)
            # Process each protein and create datasets
            for uniprot_id in uniprot_ids:
                pdb_file = self.fetch_alphafold_structure(uniprot_id)
                y = self.fetch_protein_attributes(uniprot_id)

                if pdb_file and y:
                    structure = self.parse_pdb(pdb_file)
                    residues, ca_coordinates, cb_vectors, distances = self.calculate_residue_ca_distances_and_vectors(structure)
                    edges = self.calculate_edges(ca_coordinates, cb_vectors, distances)

                    data = self.create_torch_geometric_data(residues, ca_coordinates, residue_map, cb_vectors, edges, y)
                    data.id = uniprot_id
                    datasets.append(data)

            dataset = CustomDataset(datasets, self.data_dir)

        elif self.parameters.data_name in ["manual"]:
            data = load_manual_graph()
            dataset = CustomDataset([data], self.data_dir)

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset


class CellComplexLoader(AbstractLoader):
    r"""Loader for cell complex datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        r"""Load cell complex dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_cell_complex_dataset(self.parameters)


class SimplicialLoader(AbstractLoader):
    r"""Loader for simplicial datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        r"""Load simplicial dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_simplicial_dataset(self.parameters)


class HypergraphLoader(AbstractLoader):
    r"""Loader for hypergraph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        r"""Load hypergraph dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_hypergraph_pickle_dataset(self.parameters)
