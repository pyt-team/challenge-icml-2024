import torch
import torch_geometric

from torch_geometric.transforms import AddLaplacianEigenvectorPE, SVDFeatureReduction

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
        _verify_cover_parameters(resolution, cover)
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

        lower_endpoints = torch.linspace(data_min, data_max-cover_width, self.resolution+1) 

        
        upper_endpoints = lower_endpoints+cover_width
        self.left_endpoints = lower_endpoints
        self.right_endpoints = upper_endpoints
        
        # print(torch.stack([lower_endpoints, upper_endpoints]))
        
        lower_values = torch.ge(filtered_data, lower_endpoints) # want a n x resolution Boolean tensor

        upper_values = torch.le(filtered_data, upper_endpoints) # want a n x resolution Boolean tensor 

        mask = torch.logical_and(lower_values,upper_values)
        
        return mask

    @staticmethod
    def _verify_cover_parameters(resolution, gain):
        assert gain > 0 and gain <= 0.5,/
        f"Gain must be a proportion greater than 0 and at most 0.5. Currently, gain is {gain}."
        assert resolution > 0, f"Resolution should be greater than 0. Currently,
        resolution is {resolution}."
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
        _verify_filter_parameters(filter_attr, filter_func)
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
        if self.filter_attr in filter_dict.keys():
            transform = filter_dict[self.filter_attr]
            transformed_data = transform(data)
            if self.filter_attr == "laplacian":
                filtered_data = transformed_data["laplacian_eigenvector_pe"]
            if self.filter_attr == "svd":
                filtered_data = transformed_data.x
            if self.filter_attr == "pca":
                filtered_data = torch.matmul(data.pos,
                                             transformed_data[2][:, :1]
                                            )
            else:
                filtered_data = transformed_data
        else:
            transform = self.filter_func
            filtered_data = transform(data)
        assert filtered_data.size[1] == 1, f'filtered data should have size [n_samples, 1]. Currently filtered data has size {filtered_data.size}.'
        self.filtered_data = {self.filter_attr : filtered_data}
        return filtered_data

    def _cluster(self, cover_mask):
        return None

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
        
        return {"incidence_hyperedges": a,
                "num_hyperedges": b,
                "x_0": c
                
    @staticmethod
    def _verify_filter_parameters(filter_attr, filter_func):
        filter_attr_type = type(filter_attr)
        assert (filter_attr_type is str or filter_attr is None), f"filter_attr must be a string or None."
        if filter_func is None:
            assert filter_attr in filter_dict.keys(),/
            f"Please add function to filter_func or choose filter_attr from {list(filter_dict.keys())}. Currently filter_func is {filter_func} and filter_attr is {filter_attr}."