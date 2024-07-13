import math
from itertools import combinations
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from toponetx.classes.simplicial_complex import SimplicialComplex
from torch import Tensor
from torch.nn import Linear, Parameter, init
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import scatter, softmax
from torch_sparse import SparseTensor
from tqdm import tqdm

from modules.data.utils.utils import get_complex_connectivity

from .base import PointCloud2SimplicialLifting


class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.MAX = MAX
        self._num_timescales = d_model

    def forward(self, P_time):
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)
        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1
        )  # T x B x d_pe
        if pe.is_sparse:
            pe = pe.to_dense()

        pe = pe.type(torch.FloatTensor).permute(1, 0, 2, 3)
        return torch.cat([pe, pe], dim=1)


class Raindrop(MessagePassing):
    _alpha: OptTensor = None

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        n_nodes: int,
        ob_dim: int,
        incidence_matrix: Tensor,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)
        self.process_inc_mat(incidence_matrix)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self.n_nodes = n_nodes
        self.nodewise_weights = Parameter(
            torch.Tensor(self.n_nodes, heads * out_channels)
        )

        self.increase_dim = Linear(in_channels[1], heads * out_channels * 8)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))

        self.ob_dim = ob_dim
        self.index = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        glorot(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        p_t: Tensor,
        edge_index: Adj,
        # edge_weights=None,
        use_beta=False,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        """Here, the edge_attr is not edge weights, but edge features!
        If we want to the calculation contains edge weights, change the calculation of alpha"""

        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(
            edge_index,
            x=x,
            edge_weights=self.attn_matrix,
            edge_attr=edge_attr,
            size=None,
        )

        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    def process_inc_mat(self, incidence_matrix) -> None:
        self._attn_matrix = torch.Tensor(
            incidence_matrix,
        )

    @property
    def attn_matrix(self):
        return self._attn_matrix


class DropTop(nn.Module):
    def __init__(
        self,
        n_sens,
        d_model,
        n_layers,
        n_heads,
        drop_ratio,
        n_classes,
        d_static,
        max_len=215,
        threshold=0.2,
    ):
        super(DropTop, self).__init__()

        self.d_model = d_model
        self.n_sens = n_sens
        self.d_ob = self.d_model // n_sens
        self.d_pe = 1
        self.threshold = threshold

        self.dropout = nn.Dropout(p=drop_ratio)
        self.emb = nn.Linear(d_static, n_sens)
        self.pos_encoder = PositionalEncodingTF(self.d_pe)

        self.n_combs_01 = len(list(combinations(range(n_sens), 2)))
        self.n_combs_02 = len(list(combinations(range(n_sens), 3)))
        self.n_combs_12 = len(list(combinations(range(self.n_combs_01), 3)))

        self.R_u01 = nn.Parameter(torch.Tensor(n_sens, self.n_combs_01))
        self.R_u02 = nn.Parameter(torch.Tensor(n_sens, self.n_combs_02))
        self.R_u12 = nn.Parameter(torch.Tensor(self.n_combs_01, self.n_combs_12))

        self.dropout = nn.Dropout(p=drop_ratio)

        self.inc_mat_01 = torch.zeros(n_sens, self.n_combs_01)
        self.inc_mat_02 = torch.zeros(n_sens, self.n_combs_02)
        self.inc_mat_12 = torch.zeros(self.n_combs_01, self.n_combs_12)
        self.create_simplex_strucure(n_sens)

        self.RD01 = Raindrop(
            max_len,
            max_len,
            heads=2,
            incidence_matrix=self.inc_mat_01,
            n_nodes=n_sens,
            ob_dim=self.d_ob,
        )
        self.RD02 = Raindrop(
            max_len,
            max_len,
            heads=2,
            incidence_matrix=self.inc_mat_02,
            n_nodes=n_sens,
            ob_dim=self.d_ob,
        )
        self.RD12 = Raindrop(
            max_len,
            max_len,
            heads=2,
            incidence_matrix=self.inc_mat_12,
            n_nodes=self.n_combs_12,
            ob_dim=self.d_ob,
        )

        encoder_layers = nn.TransformerEncoderLayer(
            2 * self.d_model + self.d_pe, n_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.fin_dim = 2 * self.d_model + self.d_pe + self.n_sens

        self.mlp = nn.Sequential(
            nn.Linear(self.fin_dim, self.fin_dim),
            nn.ReLU(),
            nn.Linear(self.fin_dim, n_classes),
        )

    def forward(self, x, times, static):
        self.update_simplicial_complex()
        N, B, S = x.shape
        pe = self.pos_encoder(times).permute(0, 1, 3, 2).squeeze()
        emb = self.emb(static)

        x01 = self.dropout(F.relu(x @ self.R_u01))
        x02 = self.dropout(F.relu(x @ self.R_u02))
        x = x.reshape(N * B, S)
        x1 = x @ self.inc_mat_01
        x1 = x1.reshape(N, B, -1)
        x12 = self.dropout(F.relu(x1 @ self.R_u12))

        # computes edge indexes
        edge_index_01 = torch.nonzero(self.inc_mat_01 > self.threshold).T
        edge_index_02 = torch.nonzero(self.inc_mat_02 > self.threshold).T
        edge_index_12 = torch.nonzero(self.inc_mat_12 > self.threshold).T

        out01 = torch.zeros(self.n_combs_01 // self.d_ob, B, N * self.d_ob)
        out02 = torch.zeros(self.n_combs_02 // self.d_ob, B, N * self.d_ob)
        out12 = torch.zeros(self.n_combs_12 // self.d_ob, B, N * self.d_ob)

        for unit in range(B):
            x01_u = self.select_batch_unit(x01, unit).reshape(self.n_combs_01, N)
            x02_u = self.select_batch_unit(x02, unit).reshape(self.n_combs_02, N)
            x12_u = self.select_batch_unit(x12, unit).reshape(self.n_combs_12, N)
            out01[:, unit, :] = self.RD01(x01_u, p_t=pe, edge_index=edge_index_01)
            out02[:, unit, :] = self.RD02(x02_u, p_t=pe, edge_index=edge_index_02)
            out12[:, unit, :] = self.RD12(x12_u, p_t=pe, edge_index=edge_index_12)

        out01 = out01.permute(1, 2, 0)
        out02 = out02.permute(1, 2, 0)
        out12 = out12.permute(1, 2, 0)

        output = torch.cat([out01, out02, out12, pe], dim=-1)
        output = self.transformer_encoder(output)

        output = torch.sum(output, dim=1)
        output = torch.cat([output, emb], dim=-1)
        output = self.mlp(output)
        return output

    def select_batch_unit(self, x, unit):
        return x[:, unit, :]

    def update_simplicial_complex(
        self,
    ) -> None:
        self.inc_mat_01 = self.RD01.attn_matrix.clone().detach()
        self.inc_mat_02 = self.RD02.attn_matrix.clone().detach()
        self.inc_mat_12 = self.RD12.attn_matrix.clone().detach()

    @classmethod
    def create_complete_simplex(cls, n_sens) -> SimplicialComplex:
        """
        Create a complete simplicial complex of n_sens nodes
        """
        sc = SimplicialComplex()
        sc.add_simplices_from([(i,) for i in range(n_sens)])
        sc.add_simplices_from(
            [(i, j) for i in range(n_sens) for j in range(i + 1, n_sens)]
        )
        sc.add_simplices_from(
            [
                (i, j, k)
                for i in range(n_sens)
                for j in range(i + 1, n_sens)
                for k in range(j + 1, n_sens)
            ]
        )
        return sc

    def create_simplex_strucure(self, n_sens) -> None:
        """
        Create the incidence matrices for a complete simplicial complex of order 2 on n_sens nodes (0-cells)
        """
        sc = self.create_complete_simplex(n_sens)
        self.zero_cells = sc.skeleton(0)
        self.one_cells = sc.skeleton(1)
        self.two_cells = sc.skeleton(2)
        for i in range(len(self.zero_cells)):
            for j in range(len(self.one_cells)):
                if set(self.zero_cells[i]).issubset(set(self.one_cells[j])):
                    self.inc_mat_01[i, j] = 1

        for i in range(len(self.zero_cells)):
            for j in range(len(self.two_cells)):
                if set(self.zero_cells[i]).issubset(set(self.two_cells[j])):
                    self.inc_mat_02[i, j] = 1

        for i in range(len(self.one_cells)):
            for j in range(len(self.two_cells)):
                if set(self.one_cells[i]).issubset(set(self.two_cells[j])):
                    self.inc_mat_12[i, j] = 1


class RaindropDropTopLifting(PointCloud2SimplicialLifting):
    def __init__(
        self,
        n_sens,
        d_model,
        n_layers,
        n_heads,
        drop_ratio,
        n_classes,
        d_static,
        max_len=215,
        threshold=0.2,
        epoch=1,
        batch_size=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.drop_top = DropTop(
            n_sens,
            d_model,
            n_layers,
            n_heads,
            drop_ratio,
            n_classes,
            d_static,
            max_len,
            threshold,
        )
        self.optimizer = torch.optim.Adam(self.drop_top.parameters(), lr=0.01)
        self.epoch = epoch
        self.batch_size = batch_size

    def _get_lifted_topology(self, sc: SimplicialComplex) -> dict:
        lifted_topology = get_complex_connectivity(sc, self.complex_dim)

        lifted_topology["x_0"] = torch.stack(
            list(sc.get_simplex_attributes("weight", 0).values())
        )
        lifted_topology["x_1"] = torch.stack(
            list(sc.get_simplex_attributes("weight", 1).values())
        )
        # lifted_topology["x_2"] = torch.stack(
        #     list(sc.get_simplex_attributes("weight", 2).values())
        # )
        lifted_topology["eho"] = "bite"
        print(sc.get_simplex_attributes("weight", 1), "oui")
        return lifted_topology

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        for _ in tqdm(range(self.epoch)):
            for i in range(0, data.x.shape[0], self.batch_size):
                x, t, s, y = (
                    data.x[:, i : i + self.batch_size, :],
                    data.time[:, i : i + self.batch_size],
                    data.static[i : i + self.batch_size, :],
                    data.y[i : i + self.batch_size],
                )
                self.optimizer.zero_grad()
                out = self.drop_top(x, t, s)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()

        self.drop_top.update_simplicial_complex()
        incidence_matrices = (
            self.drop_top.inc_mat_01,
            self.drop_top.inc_mat_02,
            self.drop_top.inc_mat_12,
        )
        sc = self.create_simplex(incidence_matrices)
        sc.set_simplex_attributes(
            {(i,): torch.mean(data.x[:, i, :], axis=0) for i in range(data.x.shape[1])},
            "weight",
        )
        self.complex_dim = sc.maxdim
        print(f"{sc.maxdim=}")
        return self._get_lifted_topology(sc)

    def create_simplex(self, incidence_matrices, threshold=0.2):
        inc_01, inc_02, inc_12 = incidence_matrices
        complete_sc = DropTop.create_complete_simplex(inc_01.shape[0])
        one_cells = complete_sc.skeleton(1)
        two_cells = complete_sc.skeleton(2)
        sc = SimplicialComplex()
        for i in range(inc_01.shape[0]):
            sc.add_simplex((i,))

        for i, edge in enumerate(one_cells):
            nonzero = torch.nonzero(inc_01[:, i]).T
            if all(inc_01[nonzero, i].squeeze() > threshold):
                sc.add_simplex((edge))
                sc.set_simplex_attributes(
                    {edge: torch.mean(inc_01[nonzero, i].squeeze())},
                    "weight",
                )
        for i, face in enumerate(two_cells):
            nonzero = torch.nonzero(inc_02[:, i]).T
            if all(inc_02[nonzero, i].squeeze() > threshold):
                sc.add_simplex((face))
                sc.set_simplex_attributes(
                    {face: torch.mean(inc_02[nonzero, i].squeeze())},
                    "weight",
                )

        for i, face in enumerate(two_cells):
            nonzero = torch.nonzero(inc_12[:, i]).T
            if all(inc_12[nonzero, i].squeeze() > threshold):
                sc.add_simplex((face))
                sc.set_simplex_attributes(
                    {face: torch.mean(inc_12[nonzero, i].squeeze())},
                    "weight",
                )

        return sc
