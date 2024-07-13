import os
import zipfile

import numpy as np
import requests
import rootutils
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data

from modules.data.load.base import AbstractLoader
from modules.data.utils.concat2geometric_dataset import ConcatToGeometricDataset
from modules.data.utils.custom_dataset import CustomDataset
from modules.data.utils.utils import (
    load_cell_complex_dataset,
    load_hypergraph_pickle_dataset,
    load_manual_graph,
    load_simplicial_dataset,
    preprocess_data,
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
    r"""Loader for point cloud datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(self) -> torch_geometric.data.Dataset:
        """Load point cloud dataset.

        Parameters
        ----------
        None

        Returnss
        ----------
        torch.data.Dataset
            torch.data.Dataset object containing the loaded data.
        """
        data_url = "https://figshare.com/ndownloader/files/34683085"

        root_folder = rootutils.find_root()
        data_path = os.path.join(
            root_folder, self.parameters["data_dir"], "..", "point_cloud_data.zip"
        )

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        response = requests.get(data_url)

        with open(data_path, "wb") as f:
            f.write(response.content)

        # Extract the zip file
        if data_path.endswith(".zip"):
            with zipfile.ZipFile(data_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(data_path))

        # Load the dataset
        data_file = os.path.join(
            root_folder, self.parameters["data_dir"], self.parameters["data_file"]
        )
        label_file = os.path.join(
            root_folder, self.parameters["data_dir"], self.parameters["label_file"]
        )

        np_dataset = np.load(data_file, allow_pickle=True)
        np_labels = np.load(label_file)
        y = np_labels[:, -1].reshape((-1, 1))
        x = np_dataset

        T, F = x[0]["arr"].shape
        D = len(x[0]["extended_static"])
        P = np.zeros((len(x), T, F))
        Pstatic = np.zeros((len(x), D))
        for i in range(x.shape[0]):
            P[i] = x[i]["arr"]
            Pstatic[i] = x[i]["extended_static"]

        P, Ptimes, Pstatic, y = preprocess_data(x, P, Pstatic, y)

        P = P.permute(1, 0, 2)
        Ptimes = Ptimes.permute(1, 0, 2)
        P = P[:, :, :4]

        return Data(x=P, y=y, time=Ptimes, static=Pstatic)
