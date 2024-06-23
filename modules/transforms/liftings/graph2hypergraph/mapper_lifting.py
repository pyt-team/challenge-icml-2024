import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting

class MapperCover():
    def __init__():
        self.resolution = resolution
        self.gain = gain

    def fit_transform(self, data):
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
        if self.projection_attr = None
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