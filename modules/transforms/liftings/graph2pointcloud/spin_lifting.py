import math

import torch_geometric

from modules.transforms.liftings.graph2pointcloud.base import Graph2PointcloudLifting


class SpinLifting(Graph2PointcloudLifting):
    r"""Lifts graphs to point clouds domain by placing the nodes in a rotational manner

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def find_neighbors(graph, node):
        return list(graph.neighbors(node))

    @staticmethod
    def calculate_coords_delta(angle):
        radians = math.radians(angle)
        x_delta = math.cos(radians)
        y_delta = math.sin(radians)
        return x_delta, y_delta

    def assign_coordinates(self, center_coords, neighbors):
        coords_dict = {}
        angle_to_rotate = 30
        current_angle = 0
        for neighbor in neighbors:
            if current_angle >= 360:
                angle_to_rotate /= 2
                current_angle = angle_to_rotate
            delta = self.calculate_coords_delta(current_angle)
            new_coords = (center_coords[0] + delta[0], center_coords[1] + delta[1])
            while new_coords in coords_dict.values():
                current_angle += angle_to_rotate
                if current_angle >= 360:
                    angle_to_rotate /= 2
                    current_angle = angle_to_rotate
                delta = self.calculate_coords_delta(current_angle)
                new_coords = (center_coords[0] + delta[0], center_coords[1] + delta[1])
            coords_dict[neighbor] = new_coords
            current_angle += angle_to_rotate

        return coords_dict

    def lift(self, coords, graph, start_node):
        old_coords = coords.copy()
        neighbors = self.find_neighbors(graph, start_node)
        coords.update(self.assign_coordinates(coords[start_node], neighbors))
        # Do a breadth-first traversal of the remaining nodes
        queue = neighbors
        visited = set(neighbors)
        while queue:
            current_center = queue.pop(0)
            neighbors = self.find_neighbors(graph, current_center)
            # Remove neighbors that have coordinates already assigned
            neighbors = [neighbor for neighbor in neighbors if neighbor not in coords]
            coords.update(self.assign_coordinates(coords[current_center], neighbors))
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

        # Get the new coordinates generated
        new_coords = {start_node: coords[start_node]}
        new_coords.update(
            {key: value for key, value in coords.items() if key not in old_coords}
        )
        # Find the max distance between the nodes in new_coords,
        # which will be used as the separation distance between the disconnected parts of the graph
        max_distance = 0
        for node1 in new_coords:
            for node2 in new_coords:
                distance = math.sqrt(
                    (new_coords[node1][0] - new_coords[node2][0]) ** 2
                    + (new_coords[node1][1] - new_coords[node2][1]) ** 2
                )
                if distance > max_distance:
                    max_distance = distance

        return coords, max_distance

    def lift_topology(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Lifts the topology of a graph to a point cloud by placing the nodes in a rotational manner

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted point cloud, with node names as keys and coordinates as values.
        """
        graph = self._generate_graph_from_data(data)
        coords = {}
        node_list = list(graph.nodes)
        # Assign the first node to (0, 0)
        start_node = node_list[0] if self.start_node is None else self.start_node
        coords[start_node] = (0.0, 0.0)
        # Then spin around to assign coords to its neighbors
        coords, max_distance = self.lift(coords, graph, start_node)

        # If it's a graph with multiple disconnected parts, do the above for each part
        remaining_nodes = set(node_list) - coords.keys()
        max_separation_distance = max_distance
        while remaining_nodes:
            start_node = remaining_nodes.pop()
            last_assigned_node = list(coords.keys())[-1]
            new_start_coords = (
                coords[last_assigned_node][0] + max_separation_distance,
                coords[last_assigned_node][1],
            )
            while new_start_coords in coords.values():
                new_start_coords = (
                    new_start_coords[0] + max_separation_distance,
                    new_start_coords[1],
                )
            coords[start_node] = new_start_coords
            coords, max_distance = self.lift(coords, graph, start_node)
            if max_distance > max_separation_distance:
                max_separation_distance = max_distance
            remaining_nodes = set(node_list) - coords.keys()

        return self._get_lifted_topology(coords)
