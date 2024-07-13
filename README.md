## WeightedCliqueLifting
WeightedCliqueLifting is a Python class for lifting a weighted-directed graph into a simplicial complex. This transformation is useful in topological data analysis, where higher-order interactions are represented by simplicial complexes.

# A Note:
We were unable to fully run our lifting as the Jupyter Notebook kernel would crash while verifying the simplicial complex due to a lack of sufficient memory on any of our systems. The weighted-directed graph is too large for us to run, but we hope someone else can. We have faith that our lifting from a weighted-directed graph into a simplicial complex works but would greatly appreciate it if this could be tested and we could receive feedback. Our code is commented in detail for you to work with.

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

## Relevant Information
- The deadline is **July 12th, 2024 (Anywhere on Earth)**. Participants are welcome to modify their submissions until this time.
- Please, check out the [main webpage of the challenge](https://pyt-team.github.io/packs/challenge.html) for the full description of the competition (motivation, submission requirements, evaluation, etc.)

## Brief Description
The main purpose of the challenge is to further expand the current scope and impact of Topological Deep Learning (TDL), enabling the exploration of its applicability in new contexts and scenarios. To do so, we propose participants to design and implement lifting mappings between different data structures and topological domains (point-clouds, graphs, hypergraphs, simplicial/cell/combinatorial complexes), potentially bridging the gap between TDL and all kinds of existing datasets.


## General Guidelines
Everyone can participate and participation is free --only principal PyT-Team developers are excluded. It is sufficient to:
- Send a valid Pull Request (i.e. passing all tests) before the deadline.
- Respect Submission Requirements (see below).

Teams are accepted, and there is no restriction on the number of team members. An acceptable Pull Request automatically subscribes a participant/team to the challenge.

We encourage participants to start submitting their Pull Request early on, as this helps addressing potential issues with the code. Moreover, earlier Pull Requests will be given priority consideration in the case of multiple submissions of similar quality implementing the same lifting.

A Pull Request should contain no more than one lifting. However, there is no restriction on the number of submissions (Pull Requests) per participant/team.

## Basic Setup
To develop on your machine, here are some tips.

First, we recommend using Python 3.11.3, which is the python version used to run the unit-tests.

For example, create a conda environment:
   ```bash
   conda create -n topox python=3.11.3
   conda activate topox
   ```

Then:

1. Clone a copy of tmx from source:

   ```bash
   git clone git@github.com:pyt-team/challenge-icml-2024.git
   cd challenge-icml-2024
   ```

2. Install tmx in editable mode:

   ```bash
   pip install -e '.[all]'
   ```
   **Notes:**
   - Requires pip >= 21.3. Refer: [PEP 660](https://peps.python.org/pep-0660/).
   - On Windows, use `pip install -e .[all]` instead (without quotes around `[all]`).

4. Install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.

      ```bash
      pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
      pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
      pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
      ```

      where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).

5. Ensure that you have a working tmx installation by running the entire test suite with

   ```bash
   pytest
   ```

    In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

6. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Questions

Feel free to contact us through GitHub issues on this repository, or through the [Geometry and Topology in Machine Learning slack](https://tda-in-ml.slack.com/join/shared_invite/enQtOTIyMTIyNTYxMTM2LTA2YmQyZjVjNjgxZWYzMDUyODY5MjlhMGE3ZTI1MzE4NjI2OTY0MmUyMmQ3NGE0MTNmMzNiMTViMjM2MzE4OTc#/). Alternatively, you can contact us via mail at any of these accounts: guillermo.bernardez@upc.edu, lev.telyatnikov@uniroma1.it.
