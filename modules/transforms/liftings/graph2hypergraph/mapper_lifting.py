import networkx as nx
import torch
import torch_geometric
from torch_geometric.transforms import AddLaplacianEigenvectorPE, SVDFeatureReduction
from torch_geometric.utils import subgraph

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class MapperCover:
    r"""The MapperCover class computes the cover used in constructing the Mapper
    for the MapperLifting class.

    Parameters
    ---------
    resolution : int, optional
        The number of intervals in the MapperCover. Default is 10.
    gain : float, optional
        The percentage of overlap between consectutive intervals
        in MapperCover and should be value between 0 and 0.5.
        Default is 0.3.

    Attributes
    ----------
    left_endpoints : (resolution, 1) Tensor
        The left endpoints for each interval in the MapperCover.
    right_endpoints : (resolution, 1) Tensor
        The right endpoints for each interval in the MapperCover.
    """

    def __init__(self, resolution=10, gain=0.3):
        self.resolution = resolution
        self.gain = gain
        self._verify_cover_parameters()

    def fit_transform(self, filtered_data):
        r"""Constructs an interval cover over filtered data.

        Parameters
        ----------
        filtered_data : torch_geometric.data.Data or torch.Tensor
        with size (n_sample, 1).

        Returns
        -------
         < (n_sample, resolution) boolean Tensor.
            Mask which identifies which data points are
            in each cover set. Covers which are empty
            are removed so k = number of nonempty cover sets.
        """

        data_min = torch.min(filtered_data)
        data_max = torch.max(filtered_data)
        data_range = torch.max(filtered_data) - torch.min(filtered_data)
        # width of each interval in the cover
        cover_width = data_range / (self.resolution - (self.resolution - 1) * self.gain)
        lower_endpoints = torch.linspace(
            data_min, data_max - cover_width, self.resolution + 1
        )
        upper_endpoints = lower_endpoints + cover_width
        self.left_endpoints = lower_endpoints
        self.right_endpoints = upper_endpoints
        # want a n x resolution Boolean tensor
        lower_values = torch.ge(filtered_data, lower_endpoints)
        upper_values = torch.le(filtered_data, upper_endpoints)
        mask = torch.logical_and(lower_values, upper_values)
        # remove empty intervals from cover
        non_empty_covers = torch.any(mask, 0)
        return mask[:, non_empty_covers]

    def _verify_cover_parameters(self):
        assert (
            self.gain > 0 and self.gain <= 0.5
        ), f"Gain must be a proportion greater than 0 and at most 0.5. Currently, gain is {self.gain}."
        assert (
            self.resolution > 0
        ), f"Resolution should be greater than 0. Currently, resolution is {self.resolution}."
        assert float(
            self.resolution
        ).is_integer(), f"Resolution must be an integer value. Currenly, resolution is {self.resolution}."


# Global filter dictionary for the MapperLifting class.
filter_dict = {
    "laplacian": AddLaplacianEigenvectorPE(k=1),
    "svd": SVDFeatureReduction(out_channels=1),
    "pca": lambda data: torch.pca_lowrank(data.pos, q=1),
    "feature_sum": lambda data: torch.sum(data.x, dim=1),
    "position_sum": lambda data: torch.sum(data.pos, dim=1),
}


class MapperLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain using a Mapper construction for CC-pooling.
    (See Figure 30 in [1])

    Parameters
    ----------
    filter_attr : str, optional
        Name of the filter functional to filter data to 1-dimensional subspace.
        The filter attribute can be "laplacican", "svd", "pca", "feature_sum",
        "position_sum". You may also define your own filter_attr string if
        the filter_func parameter is defined.
        Default is "laplacian".
    resolution : int, optional
        The number of intervals to construct the MapperCover.
        Default is 10.
    gain : float, optional
        The percentage of overlap between consectutive intervals
        in MapperCover and should be value between 0 and 0.5.
        Default is 0.3.
    filter_func : object, optional
        Filter function used for Mapper construction.
        Self defined lambda function or transform to filter data.
        Function must output an (n_sample, 1) Tensor.
        If filter_func is not None, user must define filter_attr
        as a string not already listed above.
        Default is None.
    **kwargs : optional
        Additional arguments for the class.

    Notes
    -----
    The following are common filter functions which can be called with
    filter_attr.

    1. "laplacian" : Applies the torch_geometric.transforms.AddLaplacianEigenvectorPE(k=1)
    transform and projects onto the 1st eigenvector.

    2. "svd" : Applies the torch_geometric.transforms.SVDFeatureReduction(out_channels=1)
    transform to project to 1-dimensional subspace.

    3. "pca" : Applies torch.pca_lowrank(q=1) transform and then projects to the 1st
    principle component.

    4. "feature_sum" : Applies torch.sum(dim=1) to the feature space of nodes in the graph
    (ie. torch_geometric.Data.data.x).

    5. "position_sum" : Applies torch.sum(dim=1) to the position of nodes in the graph
    (ie. torch_geometric.Data.data.pos).

    You may also construct your own filter_attr and filter_func:

    6. "my_filter_attr" : my_filter_func = lambda data : my_filter_func(data)
    where my_filter_func(data) outputs a (n_sample, 1) Tensor.

    References
    ----------
    .. [1] Hajij, M., Zamzmi, G., Papamarkou, T., Miolane, N., Guzmán-Sáenz,
        A., Ramamurthy, K. N., et al. (2022).
        Topological deep learning: Going beyond graph data.
        arXiv preprint arXiv:2206.00606.
    """

    def __init__(
        self,
        filter_attr="laplacian",
        resolution=10,
        gain=0.3,
        filter_func=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filter_attr = filter_attr
        self.resolution = resolution
        self.gain = gain
        self.filter_func = filter_func
        self._verify_filter_parameters()

    def _filter(self, data):
        """Applies 1-dimensional filter function to
        torch_geometric.Data.data.
        """
        if self.filter_attr in filter_dict:
            transform = filter_dict[self.filter_attr]
            transformed_data = transform(data)
            if self.filter_attr == "laplacian":
                filtered_data = transformed_data["laplacian_eigenvector_pe"]
            if self.filter_attr == "svd":
                filtered_data = transformed_data.x
            if self.filter_attr == "pca":
                filtered_data = torch.matmul(data.pos, transformed_data[2][:, :1])
            if self.filter_attr not in ["laplacian", "svd", "pca"]:
                filtered_data = transformed_data

        else:
            transform = self.filter_func
            filtered_data = transform(data)

        assert filtered_data.size() == torch.Size(
            [len(data.x), 1]
        ), f"filtered data should have size [n_samples, 1]. Currently filtered data has size {filtered_data.size()}."
        self.filtered_data = {self.filter_attr: filtered_data}
        return filtered_data

    def _cluster(self, data, cover_mask):
        """Finds clusters in each cover set within cover_mask.
        For each cover set, a cluster is a
        distinct connected component.
        Clusters are stored in dictionary, self.clusters.
        """
        mapper_clusters = {}
        num_clusters = 0
        # Each cover set is of the form [1, n_samples]
        for i, cover_set in enumerate(cover_mask.T):
            # Find indices of nodes which are in each cover set
            # cover_data = data.subgraph(cover_set.T) does not work
            # as it relabels node indices
            cover_data, _ = torch_geometric.utils.subgraph(
                torch.transpose(cover_set, 0, 1), data["edge_index"]
            )
            edges = [
                (i.item(), j.item())
                for i, j in zip(cover_data[0], cover_data[1], strict=False)
            ]
            if data.is_undirected():
                cover_graph = nx.Graph()
                cover_graph.add_edges_from(edges)
                # find clusters
                clusters = nx.connected_components(cover_graph)
            if data.is_directed():
                cover_graph = nx.DiGraph()
                cover_graph.add_edges_from(edges)
                # find clusters
                clusters = nx.weakly_connected_components(cover_graph)

            for cluster in clusters:
                # index is the subset of nodes in data
                # contained in cluster
                index = torch.Tensor(list(cluster))
                # kth cluster is item in dictionary
                # of the form
                # k : (cover_set_index, nodes_in_cluster)
                mapper_clusters[num_clusters] = (i, index)
                num_clusters += 1

        self.clusters = mapper_clusters
        return mapper_clusters

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain by considering k-nearest neighbors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Attributes
        ----------
        filtered_data : dict
            Filtered data used to compute the Mapper lifting.
            Dictionary is of the form
            {filter_attr: filter_func(data)}.
        cover : (n_sample, resolution) boolean Tensor
            Mask computed from the MapperCover class
            to compute the Mapper lifting.
        clusters : dict
            Distinct connected components in each cover set
            computed after fitting the Mapper cover.
            Dictionary has integer keys and tuple values
            of the form (cover_set_i, nodes_in_cluster).
            Each cluster is a rank 2 hyperedge in the
            hypergraph.

        Returns
        -------
        dict
            The lifted topology.
        """
        # Filter the data to 1-dimensional subspace
        filtered_data = self._filter(data)
        # Define and fit the cover
        cover = MapperCover(self.resolution, self.gain)
        cover_mask = cover.fit_transform(filtered_data)
        # Find the clusters in the fitted cover
        mapper_clusters = self._cluster(data, cover_mask)
        # Construct the hypergraph dictionary
        num_nodes = data["x"].shape[0]
        num_edges = data["edge_attr"].size()[0]
        num_clusters = len(mapper_clusters)
        num_hyperedges = num_edges + num_clusters

        incidence_1_edges = torch.zeros(num_nodes, num_edges)

        for i, edge in enumerate(torch.t(data["edge_index"])):
            incidence_1_edges[edge[0], i] = 1
            incidence_1_edges[edge[1], i] = 1

        incidence_1_hyperedges = torch.zeros(num_nodes, num_clusters)

        for i, hyperedge in enumerate(mapper_clusters):
            for j in mapper_clusters[hyperedge][1]:
                incidence_1_hyperedges[j.int(), i] = 1
        # Incidence matrix is (num_nodes, num_edges + num_clusters) size matrix
        incidence_1 = torch.hstack([incidence_1_edges, incidence_1_hyperedges])
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()

        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }

    def _verify_filter_parameters(self):
        assert type(self.filter_attr) is str, f"filter_attr must be a string or None."
        if self.filter_func is None:
            assert (
                self.filter_attr in filter_dict
            ), f"Please add function to filter_func or choose filter_attr from {list(filter_dict)}. \
            Currently filter_func is {self.filter_func} and filter_attr is {self.filter_attr}."
        if self.filter_func is not None:
            assert (
                self.filter_attr not in filter_dict
            ), f"Assign new filter_attr not in {list(filter_dict)} or leave filter_func as None. \
            Currently filter_func is {self.filter_func} and filter_attr is {self.filter_attr}"
