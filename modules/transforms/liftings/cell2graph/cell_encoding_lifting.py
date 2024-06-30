import torch
import torch_geometric
from toponetx.classes import CellComplex

from modules.transforms.liftings.cell2graph.base import Cell2GraphLifting


class CellEncodingLifting(Cell2GraphLifting):
    r"""Lifts cell complex data to graph by using CellEncoding
    Parameters
    ----------
    min_dim : int
        Specify the minimum dimension of cells to encode.
    **kwargs : optional
        Additional arguments for the class
    """

    def __init__(self, min_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.min_dim = min_dim

    def lift_topology(self, cell_complex: CellComplex) -> dict:
        r"""Lifts a cell complex dataset to a graph by using CellEncoding
        as described in 'Reducing learning on cell complexes to graphs' by
        Fabian Jogl, Maximilian Thiessen and Thomas Gaertner.
        Parameters
        ----------
        data :  torch_geometric.data.Data
            The input data to be lifted
        Returns
        -------
        dict
            The lifted topology
        """
        edge_index = []
        node_features = []

        # Create a mapping from cells to indices
        cell_to_index = {cell: i for i, cell in enumerate(cell_complex.cells())}

        # Add edges according to defintion
        edge_index = [
            [cell_to_index[tau], cell_to_index[delta]]
            for tau in cell_complex.cells()
            for delta in cell_complex.cells()
            if (
                cell_complex.is_boundary(tau, delta)
                or cell_complex.is_boundary(delta, tau)
            )
            or any(
                cell_complex.is_boundary(tau, sigma)
                and cell_complex.is_boundary(delta, sigma)
                for sigma in cell_complex.cells()
            )
        ]

        # Encode cell dimension
        max_dim = max(self.min_dim + 1, cell_complex.dimension())
        for cell in cell_complex.nodes():
            dim = cell_complex.dim(cell)
            one_hot = torch.zeros(max_dim + 1)
            one_hot[dim] = 1
            node_features.append(one_hot)

        # Create torch_geometric.Data object
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_features = torch.stack(node_features, dim=0)

        data = torch_geometric.Data(x=node_features, edge_index=edge_index)

        return {
            "shape": [data.x.shape[0], data.edge_index.shape[1]],
            "edge_index": data.edge_index,
            "num_nodes": data.x.shape[0],
        }
