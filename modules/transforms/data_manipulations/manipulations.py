import torch
import torch_geometric
from torch_geometric.utils import one_hot
from typing import Dict


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


def find_non_overlap(tensor1, tensor2):
    """
    
    """

    # Initialize an output boolean tensor with the same shape as tensor2
    output_tensor = torch.zeros_like(tensor2, dtype=torch.bool)
    
    # Loop through each row
    for i, (row1, row2) in enumerate(zip(tensor1, tensor2)):
        # Convert the rows to sets
        set1 = set(row1.tolist())
        set2 = set(row2.tolist())
        
        # Find the symmetric difference (which should be a single element)
        non_overlapping_set = set1.symmetric_difference(set2)

        # There should be exactly one non-overlapping item
        if len(non_overlapping_set) == 1:
            non_overlapping_item = list(non_overlapping_set)[0]
            # Find the position of the non-overlapping item in tensor2's row
            pos = row2.tolist().index(non_overlapping_item)
            # Set the corresponding position in the output tensor to True
            output_tensor[i, pos] = True
        else:
            print(set1)
            print(set2)
            raise ValueError("There should be exactly one non-overlapping item per row.")
# Convert the result list to a tensor
    return output_tensor

def compute_invariance_r_to_r(simplices: Dict[int, torch.Tensor], pos: torch.Tensor, adj: Dict[int, torch.Tensor]):
    """ 
        Computes the invariances from r-cells to r-cells geometrical properties
        Parameters
        ----------
        simplices : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, r)
            Indices of each component of a simplex with respect to the nodes in the original graph 
        pos : torch.Tensor, shape = (n_rank_0_cells, 3)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        adj : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, n_rank_r_cells)
            Adjacency matrices :math:`H_r` mapping cells to cells via lower and upper cells.

    """
    adj_dict={}

    # for 0 dimensional simplices:
    result_matrix_0 = []
    #loop over number of simplexes (we use the iterator as an simplex_id)
    for sending_simplex_id in range(adj[0].shape[0]):
        row = adj[0][sending_simplex_id]
        #get the indexes all connected simplices in the adjecency matrix.
        connected_simplex_ids = [idx for idx, v in enumerate(row) if idx != sending_simplex_id and v != 0]
        # get connected nodes of connected simplices (redundant for dim 0)
        connected_nodes = simplices[0][connected_simplex_ids]
        # create tensor of sending node for efficient feature calculation
        node_tensor = torch.full((len(connected_nodes),1), sending_simplex_id)
        distance = torch.linalg.norm(pos[node_tensor] - pos[connected_nodes], dim=2)
        # two because area_send, area_receive
        area = torch.zeros(len(connected_nodes), 2)
        result_matrix_0.append(torch.cat((distance, area), dim=1))
    adj_dict[0] = torch.cat(result_matrix_0, dim=0)
    #for 1 dimensional simplexes
    result_matrix_1 = []
    for sending_simplex_id in range(adj[1].shape[0]):
        row = adj[1][sending_simplex_id]
        connected_simplex_ids = [idx for idx, v in enumerate(row) if idx != sending_simplex_id and v != 0] 
        # get connected nodes shape:[N,2]
        connected_nodes = simplices[1][connected_simplex_ids].squeeze(1)
        sending_simplex_nodes = simplices[1][sending_simplex_id]
        #repeated_sending_simplex_nodes is also: [N,2]
        repeated_sending_simplex_nodes = sending_simplex_nodes.repeat(len(connected_nodes), 1)
        #angle calculation between edges
        vecs = pos[simplices[1][:, 0]] - pos[simplices[1][:, 1]]
        send_vec, rec_vec = vecs[repeated_sending_simplex_nodes], vecs[connected_nodes]
        send_norm, rec_norm = torch.linalg.norm(send_vec), torch.linalg.norm(rec_vec)
        dot = torch.sum(send_vec * rec_vec)
        cos_angle = dot / (send_norm * rec_norm)
        eps = 1e-6
        #numerical stability
        angle = torch.arccos(cos_angle.clamp(-1 + eps, 1 - eps))
        # distance: 1pi - a,2pi - b,3 a - b and area
        result_row=torch.stack([
            #pi - a
            torch.linalg.norm(pos[repeated_sending_simplex_nodes[:,0]] - pos[repeated_sending_simplex_nodes[:,1]], dim=1),
            #pi - b
            torch.linalg.norm(pos[repeated_sending_simplex_nodes[:,0]] - pos[connected_nodes[:,1]], dim=1),
            #a - b
            torch.linalg.norm(pos[repeated_sending_simplex_nodes[:,1]] - pos[connected_nodes[:,1]], dim=1),
            #area sending (same as distance for edges)
            torch.linalg.norm(pos[repeated_sending_simplex_nodes[:,0]] - pos[repeated_sending_simplex_nodes[:,1]], dim=1),
            #are receiving (same as distance for edges)
            torch.linalg.norm(pos[repeated_sending_simplex_nodes[:, 0]] - pos[connected_nodes[:, 1]], dim=1),
            torch.full((len(connected_nodes),),angle)
        ], dim=1)
        result_matrix_1.append(result_row)
    adj_dict[1] = torch.cat(result_matrix_1, dim=0)

    return adj_dict

def compute_invariance_r_minus_1_to_r(simplices: Dict[int, torch.Tensor], pos: torch.Tensor, inc: Dict[int, torch.Tensor]):
    """ 
        Computes the invariances from r-cells to r-cells geometrical properties
        Parameters
        ----------
        simplices : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, r)
            Indices of each component of a simplex with respect to the nodes in the original graph 
        pos : torch.Tensor, shape = (n_rank_0_cells, 3)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        inc : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_minus_1_cells, n_rank_r_cells)
            Adjacency matrices :math:`I^1_r` with weights of map from r-cells to (r-1)-cells

    """
    inc_dict={}
    # for 0_1 dimensional simplices:
    result_matrix_0_1 = []
    #loop over num of 0 simplices
    for sending_simplex_id in range(inc[0].shape[0]):
        row = inc[0][sending_simplex_id]
        #get the indexes all connected simplices in the adjecency matrix.
        connected_simplex_ids = [simplex_id for simplex_id, v in enumerate(row) if v != 0]
        #assumption: second node is always higher index, to calculate distance from node to simplex is just difference between the two points
        connected_nodes = simplices[1][connected_simplex_ids]
        # mask = ~(connected_nodes == sending_simplex_id).any(dim=1)
        mask = connected_nodes != sending_simplex_id
        connected_nodes = connected_nodes[mask]
        # create tensor of sending node for efficient feature calculation
        zero_simplex_repeated = torch.full((len(connected_nodes),), sending_simplex_id)
        distance = torch.linalg.norm(pos[zero_simplex_repeated] - pos[connected_nodes], dim=1)
        features = distance.view(-1, 1).repeat(1, 3)
        #area of the sending node is 0.
        features[:,1] = 0
        result_matrix_0_1.append(features)
        
    inc_dict[0] = torch.cat(result_matrix_0_1, dim=0)
    result_matrix_1_2 = []
    
    for sending_simplex_id in range(inc[1].shape[0]):
        row = inc[1][sending_simplex_id]
        connected_simplex_ids = [idx for idx, v in enumerate(row) if v != 0]
        # get connected nodes shape:[N,2]
        connected_nodes = simplices[2][connected_simplex_ids]
        sending_simplex_nodes = simplices[1][sending_simplex_id]
        repeated_sending_simplex_nodes = sending_simplex_nodes.repeat(len(connected_nodes), 1)
        #we don't know in where in the triangle the node is that does not belong to the sending 1d simplex, I made a func to look it up
        a_index = find_non_overlap(repeated_sending_simplex_nodes,connected_nodes)
        p1, p2, a= pos[repeated_sending_simplex_nodes[:,0]], pos[repeated_sending_simplex_nodes[:,1]], pos[connected_nodes[a_index]]
        v1, v2, b = p1 - a, p2 - a, p1 - p2
        eps = 1e-6
        v1_n, v2_n, b_n = torch.linalg.norm(v1, dim=1), torch.linalg.norm(v2, dim=1), torch.linalg.norm(b, dim=1)
        v1_a = torch.arccos((torch.sum(v1 * b, dim=1) / (v1_n * b_n)).clamp(-1 + eps, 1 - eps))
        v2_a = torch.arccos((torch.sum(v2 * b, dim=1) / (v2_n * b_n)).clamp(-1 + eps, 1 - eps))
        b_a = torch.arccos((torch.sum(v1 * v2, dim=1) / (v1_n * v2_n)).clamp(-1 + eps, 1 - eps))
        #check if dimensions of angle are appropriate [n,2]
        angle = torch.moveaxis(torch.vstack((v1_a + v2_a, b_a)), 0, 1)
        #area
        area_2 = (torch.norm(torch.cross(pos[simplices[2][connected_simplex_ids, 0]] - pos[simplices[2][connected_simplex_ids, 1]],
                                        pos[simplices[2][connected_simplex_ids, 0]] - pos[simplices[2][connected_simplex_ids, 2]], dim=1),
                            dim=1) / 2).unsqueeze(1) 
        
        distances = torch.stack([
            torch.linalg.norm(p1 - a, dim=1)
            + torch.linalg.norm(p1 - a, dim=1),
            torch.linalg.norm(p2 - a, dim=1)
        ], dim=1)
        result_matrix_1_2.append(torch.cat((distances,area_2,angle,),dim=1))
    inc_dict[1] = torch.cat(result_matrix_1_2, dim=0)
    return inc_dict 
    