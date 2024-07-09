import csv
import sys
import time

import networkx as nx
import numpy as np
import pyflagsercount as pfc
import scipy.sparse as sp
import torch

sys.path.append("../../")
from modules.data.load.loaders import GraphLoader
from modules.data.preprocess.preprocessor import PreProcessor
from modules.transforms.liftings.graph2combinatorial.sp_lifting import \
    DirectedFlagComplex as dfc
from modules.utils.utils import (describe_data, load_dataset_config,
                                 load_model_config, load_transform_config)

dataset_name = "cocitation_cora"
dataset_config = load_dataset_config(dataset_name)
loader = GraphLoader(dataset_config)
dataset = loader.load()
describe_data(dataset)
dataset_digraph = nx.DiGraph()
dataset_digraph.add_edges_from(
    list(zip(dataset.edge_index[0].tolist(), dataset.edge_index[1].tolist()))
)
flag_complex = dfc(dataset_digraph, 2)
cells = flag_complex.complex[2]  # simplicial complexes (order 2)

qij_adj = flag_complex.qij_adj(sigmas=cells, taus=cells, q=1, i=0, j=2, chunk_size=5000)

print(qij_adj.shape)
