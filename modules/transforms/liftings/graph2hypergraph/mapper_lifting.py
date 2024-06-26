import torch
import torch_geometric
import networkx as nx

from torch_geometric.transforms import AddLaplacianEigenvectorPE, SVDFeatureReduction
from torch_geometric.utils import subgraph, to_networkx

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting

class MapperCover():
    """ The MapperCover class computes

    Attributes
    ----------
    left_endpoints : (resolution, 1) Tensor
    right_endpoints : (resolution, 1) Tensor
    """
    def __init__(self, resolution = 10, gain = 0.3):
        """ 
        Resolution: Number of intervals to cover codomain in. (Default 10) 
        Gain: Proportion of interval which overlaps with next interval on each end. 
              Gain Must be greater than 0 and less than 0.5.
        """
        # self._verify_cover_parameters(resolution, cover)
        self.resolution = resolution
        self.gain = gain

    def fit_transform(self, filtered_data):
        """Inputs data: (n x 1) Tensor of values for filter ? 
           Outputs mask: (n x resolution) boolean Tensor
            """

        data_min = torch.min(filtered_data) 
        data_max = torch.max(filtered_data)
        data_range = torch.max(filtered_data)-torch.min(filtered_data) 
        cover_width = data_range/(self.resolution - (self.resolution-1)*self.gain)
        lower_endpoints = torch.linspace(data_min, 
                                         data_max-cover_width,
                                         self.resolution+1) 
        upper_endpoints = lower_endpoints+cover_width
        self.left_endpoints = lower_endpoints
        self.right_endpoints = upper_endpoints
        lower_values = torch.ge(filtered_data, lower_endpoints) # want a n x resolution Boolean tensor
        upper_values = torch.le(filtered_data, upper_endpoints) # want a n x resolution Boolean tensor 
        mask = torch.logical_and(lower_values,upper_values)
        return mask

    def _verify_cover_parameters(self, resolution, gain):
        assert gain > 0 and gain <= 0.5, \
            f"Gain must be a proportion greater than 0 and at most 0.5. Currently, gain is {gain}."
        assert resolution > 0, f"Resolution should be greater than 0. Currently, resolution is {resolution}."
        assert float(resolution).is_integer(), f"Resolution must be an integer value. Currenly, resolution is {resolution}."
        
class MapperLifting(Graph2HypergraphLifting):
    r""" Lifts graphs to hypergraph domain using a Mapper construction and CC-pooling. See [Topological Deep Learning: Going Beyond Graph Data](https://arxiv.org/abs/2206.00606).

    Parameters
    ----------
    filter_attr : str, optional
        Explain...
    resolution : int, optional
        The number of intervals in the MapperCover. Default is 10.
    gain : float, optional
        The percentage of overlap between consectutive intervals
        in MapperCover and should be value between 0 and 0.5. 
        Default is 0.3.
    filter_func : object, optional
        Filter function used for Mapper construction. 
        Function must output an (n_sample, 1) Tensor. Default is None.

    Attributes
    ----------
    filtered_data : dict
        Filtered data used to compute the Mapper lifting. Given 
        as a dictionary {`filter_attr`: `filter_func(data)`}
    cover : MapperCover fitted to compute the Mapper lifting.
    
    

    """
    filter_dict = {
        "laplacian" : AddLaplacianEigenvectorPE(k=1),
        "svd" : SVDFeatureReduction(out_channels=1),
        "pca" : lambda data : torch.pca_lowrank(data.pos, q=1),
        "feature_sum" : lambda data : torch.sum(data.x, 1),
        "position_sum" : lambda data : torch.sum(data.pos, 1),
    }

    

    def __init__(self, 
                 filter_attr = "laplacian", 
                 resolution = 10, 
                 gain = 0.3, 
                 filter_func = None,
                 **kwargs
                ):
        
        #self._verify_filter_parameters(filter_attr, filter_func)
        super().__init__(**kwargs)
        self.filter_attr = filter_attr
        self.resolution = resolution
        self.gain = gain
        self.filter_func = filter_func 
    """

    filter_attr: laplacian, sum, svd (pca?), (lambda?)

    all of these add a node feature and then 

    just say: add an attribute to your data that you wnat to filter on. 
    then provide the key to that attribute
    exmaples; pca, laplacian, etc

    """


    
    def _filter(self, data):
        if self.filter_attr in self.filter_dict.keys():
            transform = self.filter_dict[self.filter_attr]
            transformed_data = transform(data)
            if self.filter_attr == "laplacian":
                filtered_data = transformed_data["laplacian_eigenvector_pe"]
            if self.filter_attr == "svd":
                filtered_data = transformed_data.x
            if self.filter_attr == "pca":
                filtered_data = torch.matmul(data.pos,
                                             transformed_data[2][:, :1]
                                            )
            if self.filter_attr not in ["laplacian","svd","pca"]:
                filtered_data = transformed_data
                
        else:
            transform = self.filter_func
            filtered_data = transform(data)
        
        assert filtered_data.size() == torch.Size([len(data.x),1]),\
                f'filtered data should have size [n_samples, 1]. Currently filtered data has size {filtered_data.size()}.'
        self.filtered_data = {self.filter_attr : filtered_data}
        return filtered_data

    def _cluster(self, data, cover_mask):
        """Finds clusters in each cover set and computes the hypergraph.
        """
        num_nodes = data.x.shape[0]
        mapper_clusters = {}
        num_clusters = 0
         # Each cover set is of the form [1, n_samples]
        for i, cover_set in enumerate(cover_mask.T):
            # Find indices of nodes which are in each cover set
            
            #cover_data = data.subgraph(cover_set.T, relabel_nodes=False) does not work 

            # if len(cover_set)==0:
            #     continue
            
            cover_data, _ = torch_geometric.utils.subgraph(cover_set.T, data["edge_index"])  #DATA.SUBGRAPH sets relabel_nodes to True
            
            cover_graph = nx.Graph() 

            edges = [
                (i.item(), j.item())
                for i, j in zip(cover_data[0], cover_data[1], strict=False)
                    ]
            
            #cover_graph = torch_geometric.utils.convert.to_networkx(cover_data, to_undirected = True)
            
            # if cover_data.is_directed():
            #     clusters = nx.weakly_connected_components(cover_graph)

            cover_graph.add_edges_from(edges)
            
            
            clusters = nx.connected_components(cover_graph)

            for cluster_index in clusters:
                index = torch.Tensor(list(cluster_index))
                mapper_clusters[num_clusters] = (i,index)
                num_clusters += 1
                
        return mapper_clusters

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain by considering k-nearest neighbors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        filtered_data = self._filter(data)
        cover = MapperCover(self.resolution, self.gain)
        cover_mask = cover.fit_transform(filtered_data)
        mapper_clusters = self._cluster(data, cover_mask)
        
        num_nodes = data["x"].shape[0]
        num_edges = data["edge_attr"].size()[0]
        num_clusters = len(mapper_clusters)        
        num_hyperedges = num_edges + num_clusters
        
        incidence_1_edges = torch.zeros(num_nodes, num_edges)

        for i,edge in enumerate(data["edge_index"].T): 
            incidence_1_edges[edge[0],i] = 1
            incidence_1_edges[edge[1],i] = 1

        incidence_1_hyperedges = torch.zeros(num_nodes, num_clusters)

        for i, hyperedge in enumerate(mapper_clusters): 
            for j in mapper_clusters[hyperedge][1]: 
                incidence_1_hyperedges[j.int(),i] = 1 
                
        incidence_1 = torch.hstack([incidence_1_edges, incidence_1_hyperedges])

        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()

        
        return {"incidence_hyperedges": incidence_1,
                "num_hyperedges": num_hyperedges,
                "x_0": data.x,
               }
                
    def _verify_filter_parameters(self, filter_attr, filter_func):
        filter_attr_type = type(filter_attr)
        assert (filter_attr_type is str or filter_attr is None), f"filter_attr must be a string or None."
        if filter_func is None:
            assert filter_attr in self.filter_dict.keys(), \
            f"Please add function to filter_func or choose filter_attr from {list(filter_dict)}. \
            Currently filter_func is {filter_func} and filter_attr is {filter_attr}."