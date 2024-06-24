import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting

class MapperCover():
    def __init__(self, resolution = 10, gain = 0.3):
        """ 
        Resolution: Number of intervals to cover codomain in. (Default 10) 
        Gain: Proportion of interval which overlaps with next interval on each end. 
              Gain Must be greater than 0 and less than 0.5.
        """

        assert gain > 0 and gain < 0.5, "Gain must be a proportion greater than 0 and less than 0.5."
        
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
        
        print(torch.stack([lower_endpoints, upper_endpoints]))
        
        lower_values = torch.ge(filtered_data, lower_endpoints) # want a n x resolution Boolean tensor

        upper_values = torch.le(filtered_data, upper_endpoints) # want a n x resolution Boolean tensor 

        mask = torch.logical_and(lower_values,upper_values)
        
        return mask
        
class MapperLifting(Graph2HypergraphLifting):

    def __init__(self, projection_domain = 'pos', projection_attr = None, resolution = 10, gain = 0.3, **kwargs):
        super.__init__(**kwargs)
        self.projection_domain = projection_domain
        self.projection_attr = projection_attr
        self.resolution = resoluion
        self.gain = gain
    """
    Need to construct the filter functions. Slightly confused about 
    torch_geometric.data.Data type. data.x will give feature matrix
    for nodes, but it can also have string based feature attributes?
    Maybe we should just do some sort of eccentricity filter for these
    feature matrices (both for edges and nodes). Maybe also for position?
    """
    def _filter_graph(self, data):
        verify_graph_attrs(data, self.projection_domain, self.projection_attr)
        filtered_data = data.x
        return filtered_data
        
    def _filter_pos(self, data):
        if self.projection_attr == None:
        # PCA onto 1st principle component
            _, _, V = torch.pca_lowrank(data.pos)
            filtered_data = torch.matmul(data.pos, V[:,:1])
        return filtered_data
        
    def _filter(self, data):
        if projection_domain == 'pos':
            filtered_data = self._filter_pos(data)
        if projection_domain == 'node':
            filtered_data = self._filter_graph(data)
        return filtered_data

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
        }

    @staticmethod
    def verify_parameters():
        return None
    @staticmethod
    def verify_graph_attrs(data, obj, attr):
        if obj == 'node':
            assert data.is_node_attr(attr), \
            f'{attr} is not in {obj} attributes.'
        if obj == 'edge':
            data.is_edge_attr(attr), \
            f'{attr} is not in {obj} attributes.'