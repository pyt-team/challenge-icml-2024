import networkx as nx
import torch
from toponetx.classes import CellComplex
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from modules.transforms.liftings.cell2graph.base import Cell2GraphLifting


class CellEncodingLifting(Cell2GraphLifting):
    r"""Lifts cell complex data to graph by using CellEncoding
    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encodings = {
            0: torch.tensor([1, 0, 0], dtype=torch.float),
            1: torch.tensor([0, 1, 0], dtype=torch.float),
            2: torch.tensor([0, 0, 1], dtype=torch.float),
        }

    def data2cell_complex(self, data: Data) -> CellComplex:
        r"""Helper function to transform a torch_geometric.data.Data
        object to a toponetx.classes.CellComplex object. E.g. previous
        liftings might return a Data instead of a CellComplex object.
        Parameters
        ----------
        data : torch_geometric.data.Data
            The data to be transformed to a CellComplex
        Returns
        -------
        CellComplex
            The transformed object

        """
        cc = CellComplex()

        # Add 0-cells (vertices)
        for i in range(data.num_nodes):
            cc.add_node(i)

        # Add 1-cells (edges)
        edge_index = data.edge_index.t().tolist()
        for u, v in edge_index:
            cc.add_edge(u, v)

        # Add 2-cells (faces)
        incidence_2 = data.incidence_2.t()
        for i in range(incidence_2.shape[0]):
            boundary = incidence_2[i].to_dense().nonzero().flatten().tolist()
            cc.add_cell(boundary, rank=2)

        return cc

    def lift_topology(self, cell_complex: CellComplex | Data) -> dict:
        r"""Lifts a cell complex dataset to a graph by using CellEncoding
        as described in 'Reducing learning on cell complexes to graphs' by
        Fabian Jogl, Maximilian Thiessen and Thomas Gaertner.
        Parameters
        ----------
        cell_complex :  toponetx.classes.CellComplex or torch_geometric.data.Data
            The input data to be lifted
        Returns
        -------
        dict
            The lifted topology
        """
        # Transform input to CellComplex if necessary
        if type(cell_complex) == Data:
            cell_complex = self.data2cell_complex(cell_complex)

        G = nx.Graph()

        # Add 0-cells as nodes
        G.add_nodes_from(cell_complex.nodes, cell_dim=self.encodings[0])
        G.add_edges_from(cell_complex.edges)

        # Add 1-cells
        for u, v in cell_complex.edges:
            min_e = min(u, v)
            max_e = max(u, v)
            G.add_node((min_e, max_e), cell_dim=self.encodings[1])
            G.add_edge(u, (min_e, max_e))
            G.add_edge(v, (min_e, max_e))

        # Add 2-cells
        for c in cell_complex.cells:
            G.add_node(c.elements, cell_dim=self.encodings[2])

            previous_boundaries = []
            for b0, b1 in c.boundary:
                min_b, max_b = min(b0, b1), max(b0, b1)
                G.add_edge((min_b, max_b), c.elements)
                for pre_b in previous_boundaries:
                    G.add_edge(pre_b, (min_b, max_b))
                previous_boundaries.append((min_b, max_b))

        data = from_networkx(G, group_node_attrs="cell_dim")

        return {
            "x": data.x,
            "shape": [data.x.shape[0], data.edge_index.shape[1]],
            "edge_index": data.edge_index,
            "num_nodes": data.x.shape[0],
        }
