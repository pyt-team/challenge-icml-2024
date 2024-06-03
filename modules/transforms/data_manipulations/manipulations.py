import torch
import torch_geometric
import torch_geometric.transforms
from torch_geometric.utils import one_hot
from typing import Dict

from toponetx.classes import SimplicialComplex
import gudhi

class IdentityTransform(torch_geometric.transforms.BaseTransform):
    r"""An identity transform that does nothing to the input data."""

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "domain2domain"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The (un)transformed data.
        """
        return data


class NodeFeaturesToFloat(torch_geometric.transforms.BaseTransform):
    r"""A transform that converts the node features of the input graph to float.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "map_node_features_to_float"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        data.x = data.x.float()
        return data


class NodeDegrees(torch_geometric.transforms.BaseTransform):
    r"""A transform that calculates the node degrees of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "node_degrees"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        field_to_process = []
        for key in data:
            for field_substring in self.parameters["selected_fields"]:
                if field_substring in key and key != "incidence_0":
                    field_to_process.append(key)  # noqa : PERF401

        for field in field_to_process:
            data = self.calculate_node_degrees(data, field)

        return data

    def calculate_node_degrees(
        self, data: torch_geometric.data.Data, field: str
    ) -> torch_geometric.data.Data:
        r"""Calculate the node degrees of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        field : str
            The field to calculate the node degrees.

        Returns
        -------
        torch_geometric.data.Data
        """
        if data[field].is_sparse:
            degrees = abs(data[field].to_dense()).sum(1)
        else:
            assert (
                field == "edge_index"
            ), "Following logic of finding degrees is only implemented for edge_index"
            degrees = (
                torch_geometric.utils.to_dense_adj(
                    data[field],
                    max_num_nodes=data["x"].shape[0],  # data["num_nodes"]
                )
                .squeeze(0)
                .sum(1)
            )

        if "incidence" in field:
            field_name = str(int(field.split("_")[1]) - 1) + "_cell" + "_degrees"
        else:
            field_name = "node_degrees"

        data[field_name] = degrees.unsqueeze(1)
        return data


class KeepOnlyConnectedComponent(torch_geometric.transforms.BaseTransform):
    """
    A transform that keeps only the largest connected components of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_connected_component"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        """
        Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        from torch_geometric.transforms import LargestConnectedComponents

        # torch_geometric.transforms.largest_connected_components()
        num_components = self.parameters["num_components"]
        lcc = LargestConnectedComponents(
            num_components=num_components, connection="strong"
        )
        return lcc(data)


class OneHotDegreeFeatures(torch_geometric.transforms.BaseTransform):
    r"""A transform that adds the node degree as one hot encodings to the node features.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "one_hot_degree_features"
        self.deg_field = kwargs["degrees_fields"]
        self.features_fields = kwargs["features_fields"]
        self.transform = OneHotDegree(max_degree=kwargs["max_degrees"])

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        return self.transform.forward(
            data, degrees_field=self.deg_field, features_field=self.features_fields
        )


class OneHotDegree(torch_geometric.transforms.BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features

    Parameters
    ----------
    max_degree : int
        The maximum degree of the graph.
    cat : bool, optional
        If set to `True`, the one hot encodings are concatenated to the node features.
    """

    def __init__(
        self,
        max_degree: int,
        cat: bool = False,
    ) -> None:
        self.max_degree = max_degree
        self.cat = cat

    def forward(
        self, data: torch_geometric.data.Data, degrees_field: str, features_field: str
    ) -> torch_geometric.data.Data:
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        degrees_field : str
            The field containing the node degrees.
        features_field : str
            The field containing the node features.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        assert data.edge_index is not None

        deg = data[degrees_field].to(torch.long)

        if len(deg.shape) == 2:
            deg = deg.squeeze(1)

        deg = one_hot(deg, num_classes=self.max_degree + 1)

        if self.cat:
            x = data[features_field]
            x = x.view(-1, 1) if x.dim() == 1 else x
            data[features_field] = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data[features_field] = deg

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.max_degree})"


class KeepSelectedDataFields(torch_geometric.transforms.BaseTransform):
    r"""A transform that keeps only the selected fields of the input data.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_selected_data_fields"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        # Keeps all the fields
        if len(self.parameters["keep_fields"]) == 1:
            return data

        for key, _ in data.items():  # noqa : PERF102
            if key not in self.parameters["keep_fields"]:
                del data[key]

        return data


class FilterEnoughSimplices(torch_geometric.transforms.BaseTransform):
    r"""A transform that filters out Simplicial Complexes with a maximum simplex
        dimension lower than `k`.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "filter_enough_simplices"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        pos = data.pos

        # Create a list of each node tensor position 
        points = [pos[i].tolist() for i in range(pos.shape[0])]

        # Lift the graph to an AlphaComplex
        alpha_complex = gudhi.AlphaComplex(points=points)
        simplex_tree: gudhi.SimplexTree = alpha_complex.create_simplex_tree(default_filtration_value=True)
        simplex_tree.prune_above_dimension(self.parameters["max_dim"])
        simplicial_complex = SimplicialComplex.from_gudhi(simplex_tree)

        return simplicial_complex.maxdim > 1
class InputPreproc(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "input_preproc"
        self.parameters = kwargs

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        one_hot = data.x[:, :5]
        Z_max = 9
        Z = data.x[:, 5]
        Z_tilde = (Z / Z_max).unsqueeze(1).repeat(1, 5)
        data.x = torch.cat((one_hot, Z_tilde * one_hot, Z_tilde * Z_tilde * one_hot), dim=1)

        return data

class LabelPreproc(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = 'label_preproc'
        self.parameters = kwargs

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        targets = self.parameters['targets']
        qm9_to_ev = self.parameters['qm9_to_ev']

        index = targets.index(self.parameters['target_name'])
        data.y = data.y[0, index]

        if self.parameters['target_name'] in qm9_to_ev:
            data.y *= qm9_to_ev[self.target_name]
        return data