import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import GeometricShapes
from torch_geometric.transforms import SamplePoints

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2simplicial.clique_lifting import (
    SimplicialCliqueLifting,
)
from modules.transforms.liftings.pointcloud2graph.knn_lifting import GraphKNNLifting

data = load_manual_graph()

# Initialise the class
lifting_unsigned = SimplicialCliqueLifting(complex_dim=3, signed=False)

res = lifting_unsigned.lift_topology(data.clone())

dataset = GeometricShapes(root="data/GeometricShapes")
dataset.transform = T.SamplePoints(num=256)
points = dataset[0]

knn = GraphKNNLifting(k=5)
res = knn.lift_topology(points)
