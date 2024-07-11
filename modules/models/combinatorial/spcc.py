import torch

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation
from topomodelx.nn.combinatorial.hmc_layer import HBS, HBNS


class SPCCLayer(torch.nn.Module):
    r"""Simplicial Paths Combinatorial Complex Layer

    We aim to exploit the inherent directed graph to induce higher-order motifs
    in the form of simplicial paths.

    This simple layer is only build for testing purposes: We consider a
    combinatorial complex with 0-dimensional cells (vertices), 1-dimensional
    cells (edges), 2-dimensional cells (collections of nodes contained in
    simplicial paths)

    Message passing: 0-dimensional cells (vertices) receive messages from
    0-dimensional cells (vertices) and from 2-dimensional cells (collections
    of nodes contained in simplicial paths).In the first case, adjacency
    matrices are used. In the second case, the incidence matrix from
    dimension 1 to dimension 2 is used.

    Notes
    -----
    This is a simple layer only build for testing purposes.

    Parameters
    ----------
    in_channels : list of int
        Dimension of input features on vertices (0-cells), and simplicial
        paths (3-cells). The length of the list must be 2.

    out_channels : list of int
        Dimension of output features on vertices (0-cells) and simplicial
        paths (3-cells). The length of the list must be 2.

    negative_slope : float
        Negative slope of LeakyReLU used to compute the attention
        coefficients.

    softmax_attention : bool, optional
        Whether to use softmax attention. If True, the attention
        coefficients are normalized by rows using softmax over all the
        columns that are not zero in the associated neighborhood
        matrix. If False, the normalization is done by dividing by the
        sum of the values of the coefficients in its row whose columns
        are not zero in the associated neighborhood matrix. Default is
        False.

    update_func_attention : string, optional
        Activation function used in the attention block. If None,
        no activation function is applied. Default is None.

    update_func_aggregation : string, optional
        Function used to aggregate the messages computed in each
        attention block. If None, the messages are aggregated by summing
        them. Default is None.

    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of the attention layers.
        Default is 'xavier_uniform'.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        negative_slope: float,
        softmax_attention=False,
        update_func_attention=None,
        update_func_aggregation=None,
        initialization="xavier_uniform",
    ):
        super().__init__()
        super().__init__()

        assert len(in_channels) == 2 and len(out_channels) == 2

        in_channels_0, in_channels_2 = in_channels
        out_channels_0, out_channels_2 = out_channels

        self.hbs_0 = HBS(
            source_in_channels=in_channels_0,
            source_out_channels=out_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_0_2 = HBNS(
            source_in_channels=in_channels_2,
            source_out_channels=out_channels_2,
            target_in_channels=in_channels_0,
            target_out_channels=out_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.aggr = Aggregation(aggr_func="sum", update_func=update_func_aggregation)

    def forward(self, x_0, x_2, adjacency_0, incidence_0_2):
        r"""Forward pass.

        In both message passing levels, :math:`\phi_u` and :math:`\phi_a`
        represent common activation functions within and between neighborhood
        aggregations. Both are passed to the constructor of the class as
        arguments update_func_attention and update_func_aggregation,
        respectively.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_0_cells, in_channels[0]]
            Input features on the 0-cells (vertices) of the combinatorial
            complex.
        x_2 : torch.Tensor, shape=[n_3_cells, in_channels[3]]
        Input features on the 3-cells (simplicial paths) of the combinatorial
        complex.

        adjacency_0 : torch.sparse
            shape=[n_0_cells, n_0_cells]
            Neighborhood matrix mapping 0-cells to 0-cells (A_0_up).

        incidence_0_2 : torch.sparse
            shape=[n_0_cells, n_3_cells]
            Neighborhood matrix mapping 3-cells to 0-cells (B_3).

        Returns
        -------
        _ : torch.Tensor, shape=[1, num_classes]
            Output prediction on the entire cell complex.
        """

        # Computing messages from the Simplicial Path Attention Block

        x_0_to_0 = self.hbs_0(x_0, adjacency_0)
        x_0_to_2, x_2_to_0 = self.hbns_0_2(x_2, x_0, incidence_0_2)

        x_0 = self.aggr([x_0_to_0, x_2_to_0])
        x_2 = self.aggr([x_0_to_2])

        return x_0, x_2


class SPCC(torch.nn.Module):
    """Simplicial Paths Combinatorial Complex Attention Network.

    Parameters
    ----------
    channels_per_layer : list of list of list of int
        Number of input, and output channels for each
        Simplicial Paths Combinatorial Complex Attention Layer.
        The length of the list corresponds to the number of layers.
        Each element k of the list is a list consisting of other 2
        lists. The first list contains the number of input channels for
        each input signal (nodes, sp_cells) for the k-th layer.
        The second list contains the number of output channels for
        each input signal (nodes, sp_cells) for the k-th layer.
    negative_slope : float
        Negative slope for the LeakyReLU activation.
    update_func_attention : str
        Update function for the attention mechanism. Default is "relu".
    update_func_aggregation : str
        Update function for the aggregation mechanism. Default is "relu".
    """

    def __init__(
        self,
        channels_per_layer,
        negative_slope=0.2,
        update_func_attention="relu",
        update_func_aggregation="relu",
    ) -> None:
        def check_channels_consistency():
            """Check that the number of input, and output
            channels is consistent."""
            assert len(channels_per_layer) > 0
            for i in range(len(channels_per_layer) - 1):
                assert channels_per_layer[i][2][0] == channels_per_layer[i + 1][0][0]
                assert channels_per_layer[i][2][1] == channels_per_layer[i + 1][0][1]

        super().__init__()
        check_channels_consistency()
        self.layers = torch.nn.ModuleList(
            [
                SPCCLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    negative_slope=negative_slope,
                    softmax_attention=True,
                    update_func_attention=update_func_attention,
                    update_func_aggregation=update_func_aggregation,
                )
                for in_channels, out_channels in channels_per_layer
            ]
        )

    def forward(
        self,
        x_0,
        x_2,
        neighborhood_0_to_0,
        neighborhood_0_to_2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor
            Input features on nodes.
        x_2 : torch.Tensor
            Input features on simplicial paths.
        neighborhood_0_to_0 : torch.Tensor
            Adjacency  matrix from nodes to nodes.
        neighborhood_0_to_2 : torch.Tensor
            Incidence matrix from nodes to simplicial path cells.

        Returns
        -------
        torch.Tensor, shape = (n_nodes, out_channels_0)
            Final hidden states of the nodes (0-cells).
        torch.Tensor, shape = (n_spcells, out_channels_3)
            Final hidden states of the faces (3-cells).
        """
        for layer in self.layers:
            x_0, x_2 = layer(x_0, x_2, neighborhood_0_to_0, neighborhood_0_to_2)

        return x_0, x_2


class SPCCNN(torch.nn.Module):
    """Simplicial Paths Combinatorial Complex Attention Network Model For
    Node Classification.

    Parameters
    ----------
    channels_per_layer : list of list of list of int
        Number of input, and output channels for each
        Simplicial Paths Combinatorial Complex Attention Layer.
        The length of the list corresponds to the number of layers.
        Each element k of the list is a list consisting of other 2
        lists. The first list contains the number of input channels for
        each input signal (nodes, sp_cells) for the k-th layer.
        The second list contains the number of output channels for
        each input signal (nodes, sp_cells) for the k-th layer.
    out_channels_0 : int
        Number of output channels for the 0-cells (classes)
    negative_slope : float
        Negative slope for the LeakyReLU activation.

    Returns
    -------
    torch.Tensor, shape = (n_nodes, out_channels_0)
        Final probability states of the nodes (0-cells).
    """

    def __init__(
        self,
        channels_per_layer,
        out_channels_0,
        negative_slope=0.2,
    ):
        super().__init__()
        self.base_model = SPCC(
            channels_per_layer,
            negative_slope,
        )

        self.linear = torch.nn.Linear(channels_per_layer[-1][1][0], out_channels_0)

    def forward(self, data):
        x_0 = data["x_0"]
        x_2 = data["x_2"]
        neighborhood_0_to_0 = data["adjacency_0_1"]
        neighborhood_0_to_2 = data["incidence_0_2"]

        x_0, _ = self.base_model(
            x_0,
            x_2,
            neighborhood_0_to_0,
            neighborhood_0_to_2,
        )

        x_0 = self.linear(x_0)
        return torch.softmax(x_0, dim=1)
