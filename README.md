## WeightedCliqueLifting
WeightedCliqueLifting is a Python class for lifting a weighted-directed graph into a simplicial complex by using [Ricci Curvature](https://en.wikipedia.org/wiki/Ricci_curvature) information on the graph edges. This transformation is useful in topological data analysis, where higher-order interactions are represented by simplicial complexes.

 # Table of Contents

1. Usage
2. Our New Dataset
3. Our Methods

# Usage

To use the WeightedCliqueLifting class, you need to instantiate it and call the lift_topology method with a torch_geometric.data.Data object.
```Python
import torch_geometric
from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting
from your_module import WeightedCliqueLifting

# Instantiate the class
lifting = WeightedCliqueLifting(complex_dim=3)

# Assume `data` is a torch_geometric.data.Data object
lifted_topology = lifting.lift_topology(data)
```
Please follow the tutorial in tutorials/digraph2simplicial/weighted_clique_lifting.ipynb for an example.

# Our New Dataset:
We have added a new dataset for our lifting. This dataset is an Ethereum Token Network exchange graph for September 30, 2023.

The dataset has the following attributes:
- edge_index: The indexes for all edges in the graph
- num_nodes : The number of nodes in the graph
- num_edges: the number of edges in the graph
- features: The node labels for the graph depending on what the node identifies with in the network (assets: 0, derivatives: 1, dex: 2, lending: 3, unknown: 4). Note: most are unknown, there are few nodes not labeled 4
- x: The edge weights used in our calculations explained below. The edge weights are pre-normalized for your convenience. Normalization formula taken from Dissecting Ethereum Blockchain Analytics: What We Learn from Topology and Geometry of Ethereum Graph

The dataset is stored in datasets/Transactions/EthereumTokenNetwork/EthereumTokenNetwork.pt

# Our Methods:
To perform our topological lifting from a weighted-directed graph to a simplicial complex we follow the following steps:

1. Import our dataset from torch_geometric and make it into a directed networkx graph object
2. Create a weight hashmap for the graph
3. Calculate the Forman-Ricci Curvature for the edges
4. Update the edge distances using the Forman-Ricci Curvature
5. Remove edges based on the given percentile value in the dataset config file
6. Identify cliques in the graph
7. Create a simplicial complex using the cliques


# Who We Are:
Submission Completed by Cuneyt Akcora, Ronan Buck, Vedant Patel, Guneet Uberoi from The University of Central Florida.

# ICML Topological Deep Learning Challenge 2024: Beyond the Graph Domain
Official repository for the Topological Deep Learning Challenge 2024, jointly organized by [TAG-DS](https://www.tagds.com) & PyT-Team and hosted by the [Geometry-grounded Representation Learning and Generative Modeling (GRaM) Workshop](https://gram-workshop.github.io) at ICML 2024.

 
   

## Questions

Feel free to contact us through GitHub issues on this repository, or through the [Geometry and Topology in Machine Learning slack](https://tda-in-ml.slack.com/join/shared_invite/enQtOTIyMTIyNTYxMTM2LTA2YmQyZjVjNjgxZWYzMDUyODY5MjlhMGE3ZTI1MzE4NjI2OTY0MmUyMmQ3NGE0MTNmMzNiMTViMjM2MzE4OTc#/). Alternatively, you can contact us via mail at any of these accounts: guillermo.bernardez@upc.edu, lev.telyatnikov@uniroma1.it.
