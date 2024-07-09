import os
import random

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
    load_manual_prot_pointcloud,
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


class PointCloudLoader(AbstractLoader):

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
        residue_keys = []

        for residue in residues:
            if "CA" in residue:
                ca_coord = residue["CA"].get_coord()
                residue_type = residue.get_resname()
                residue_number = residue.get_id()[1]
                key = f"{residue_type}_{residue_number}"
                ca_coordinates[key] = ca_coord
                cb_vectors[key] = residue["CB"].get_coord() - ca_coord if "CB" in residue else None
                residue_keys.append(key)

        return ca_coordinates, cb_vectors, residue_keys

    def save_point_cloud(self, ca_coordinates, cb_vectors, file_path):
        data = []
        for key, ca_coord in ca_coordinates.items():
            cb_vector = cb_vectors[key] if key in cb_vectors else np.zeros(3)
            if cb_vector is None:
                cb_vector = np.zeros(3)
            data.append({
                "residue_id": key,
                "x": ca_coord[0],
                "y": ca_coord[1],
                "z": ca_coord[2],
                "cb_x": cb_vector[0],
                "cb_y": cb_vector[1],
                "cb_z": cb_vector[2]
            })

        # Save data
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "w") as file:
            file.write("residue_id,x,y,z,cb_x,cb_y,cb_z\n")
            for row in data:
                file.write(f"{row['residue_id']},{row['x']},{row['y']},{row['z']},{row['cb_x']},{row['cb_y']},{row['cb_z']}\n")


    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        r"""Load point cloud dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """

        root_folder = rootutils.find_root()
        root_data_dir = os.path.join(root_folder, self.parameters["data_dir"])

        self.data_dir = os.path.join(root_data_dir, self.parameters["data_name"])
        if self.parameters.data_name in ["UniProt"]:
            datasets = []
            protein_data = self.fetch_uniprot_ids()
            uniprot_ids = [protein["uniprot_id"] for protein in protein_data]
            residue_map = self.residue_mapping(uniprot_ids)

            for uniprot_id in uniprot_ids:
                pdb_file = self.fetch_alphafold_structure(uniprot_id)
                y = self.fetch_protein_mass(uniprot_id)

                if pdb_file and y:
                    structure = self.parse_pdb(pdb_file)
                    ca_coordinates, cb_vectors, residue_keys = self.calculate_residue_ca_distances_and_vectors(structure)
                    point_cloud_file = os.path.join(self.data_dir, "point_cloud", f"{uniprot_id}.csv")
                    self.save_point_cloud(ca_coordinates, cb_vectors, point_cloud_file)

                    # Create one-hot residues
                    one_hot_residues = []
                    for res_id in residue_keys:
                        res_type = res_id.split("_")[0]
                        one_hot = torch.zeros(len(residue_map))
                        one_hot[residue_map[res_type]] = 1
                        one_hot_residues.append(one_hot)

                    x = torch.stack(one_hot_residues)
                    pos = torch.tensor([ca_coordinates[res_id] for res_id in residue_keys], dtype=torch.float)
                    # node_attr = [cb_vectors[res_id] for res_id in residue_keys if cb_vectors[res_id] is not None else None]
                    node_attr = []
                    for res_id in residue_keys:
                        if cb_vectors[res_id] is None:
                            node_attr.append(None)
                        else:
                            node_attr.append(cb_vectors[res_id])

                    data = torch_geometric.data.Data(
                        x=x,
                        pos=pos,
                        node_attr=node_attr,
                        y=y,
                        uniprot_id=uniprot_id
                    )

                    datasets.append(data)

            dataset = CustomDataset(datasets, self.data_dir)

        elif self.parameters.data_name in ["manual_prot"]:
            data = load_manual_prot_pointcloud()
            dataset = CustomDataset([data], self.data_dir)
        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )
        return dataset

