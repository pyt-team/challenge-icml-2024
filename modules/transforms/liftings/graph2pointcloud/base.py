import torch
import torch_geometric

from modules.transforms.liftings.lifting import GraphLifting


class Graph2PointcloudLifting(GraphLifting):
    r"""Abstract class for lifting graphs to pointclouds.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2pointcloud"
        self.start_node = kwargs.get("start_node", None)

    @staticmethod
    def _get_lifted_topology(coords: dict) -> torch_geometric.data.Data:
        r"""Returns the lifted topology.

        Parameters
        ----------
        coords : dict
            The coordinates of the nodes.

        Returns
        -------
        torch_geometric.data.Data
            The lifted topology.
        """

        # Sort the items by key to ensure correspondences between features/labels and points
        items = sorted(coords.items(), key=lambda x: x[0])
        # Convert the coordinates to tensors in order to create a torch_geometric.data.Data object
        tensor_coords = {key: torch.tensor(value) for key, value in items}
        return torch_geometric.data.Data(pos=torch.stack(list(tensor_coords.values())))
