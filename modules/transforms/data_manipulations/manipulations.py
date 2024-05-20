import torch
import torch_geometric
from torch_geometric.utils import one_hot


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

def compute_invariants_3d(feat_ind, pos, adj, inv_ind, device):
    # angles
    angle = {}

    vecs = pos[feat_ind['1'][:, 0]] - pos[feat_ind['1'][:, 1]]
    send_vec, rec_vec = vecs[adj['1_1'][0]], vecs[adj['1_1'][1]]
    send_norm, rec_norm = torch.linalg.norm(send_vec, ord=2, dim=1), torch.linalg.norm(rec_vec, ord=2, dim=1)

    dot = torch.sum(send_vec * rec_vec, dim=1)
    cos_angle = dot / (send_norm * rec_norm)
    eps = 1e-6
    angle['1_1'] = torch.arccos(cos_angle.clamp(-1 + eps, 1 - eps)).unsqueeze(1)

    # p1, p2 and a are the position of the nodes in the three endges composing it
    p1, p2, a = pos[inv_ind['1_2'][0]], pos[inv_ind['1_2'][1]], pos[inv_ind['1_2'][2]]
    # Differentece in position of the simplices
    v1, v2, b = p1 - a, p2 - a, p1 - p2

    # Norms of simplices
    v1_n, v2_n, b_n = torch.linalg.norm(v1, dim=1), torch.linalg.norm(v2, dim=1), torch.linalg.norm(b, dim=1)

    # Angles between simplices
    v1_a = torch.arccos((torch.sum(v1 * b, dim=1) / (v1_n * b_n)).clamp(-1 + eps, 1 - eps))
    v2_a = torch.arccos((torch.sum(v2 * b, dim=1) / (v2_n * b_n)).clamp(-1 + eps, 1 - eps))
    b_a = torch.arccos((torch.sum(v1 * v2, dim=1) / (v1_n * v2_n)).clamp(-1 + eps, 1 - eps))

    #TODO Figure out what is this angle
    angle['1_2'] = torch.moveaxis(torch.vstack((v1_a + v2_a, b_a)), 0, 1)

    # areas
    area = {}
    # Area of the point is zero
    area['0'] = torch.zeros(len(feat_ind['0'])).unsqueeze(1)
    # Area is the norm of the vector from p1 to p0 of the 1-simplex
    area['1'] = torch.norm(pos[feat_ind['1'][:, 0]] - pos[feat_ind['1'][:, 1]], dim=1).unsqueeze(1)

    # TODO Figure out what this area represnents
    area['2'] = (torch.norm(torch.cross(pos[feat_ind['2'][:, 0]] - pos[feat_ind['2'][:, 1]],
                                        pos[feat_ind['2'][:, 0]] - pos[feat_ind['2'][:, 2]], dim=1),
                            dim=1) / 2).unsqueeze(1)


    # Conversion to CUDA/CPU
    area = {k: v.to(feat_ind['0'].device) for k, v in area.items()}


    inv = {
        # Norm of distance between the two nodes of the 0-simplex
        '0_0': torch.linalg.norm(pos[adj['0_0'][0]] - pos[adj['0_0'][1]], dim=1).unsqueeze(1),

        '0_1': torch.linalg.norm(pos[inv_ind['0_1'][0]] - pos[inv_ind['0_1'][1]], dim=1).unsqueeze(1),
        '1_1': torch.stack([
            torch.linalg.norm(pos[inv_ind['1_1'][0]] - pos[inv_ind['1_1'][1]], dim=1),
            torch.linalg.norm(pos[inv_ind['1_1'][0]] - pos[inv_ind['1_1'][2]], dim=1),
            torch.linalg.norm(pos[inv_ind['1_1'][1]] - pos[inv_ind['1_1'][2]], dim=1),
        ], dim=1),
        '1_2': torch.stack([
            torch.linalg.norm(pos[inv_ind['1_2'][0]] - pos[inv_ind['1_2'][2]], dim=1)
            + torch.linalg.norm(pos[inv_ind['1_2'][1]] - pos[inv_ind['1_2'][2]], dim=1),
            torch.linalg.norm(pos[inv_ind['1_2'][1]] - pos[inv_ind['1_2'][2]], dim=1)
        ], dim=1),
    }

    for k, v in inv.items():
        area_send, area_rec = area[k[0]], area[k[2]]
        send, rec = adj[k]
        area_send, area_rec = area_send[send], area_rec[rec]
        inv[k] = torch.cat((v, area_send, area_rec), dim=1)

    inv['1_1'] = torch.cat((inv['1_1'], angle['1_1'].to(feat_ind['0'].device)), dim=1)
    inv['1_2'] = torch.cat((inv['1_2'], angle['1_2'].to(feat_ind['0'].device)), dim=1)

    return inv
