# """Test the message passing module."""

# import torch

# from modules.data.utils.utils import load_simple_configuration_graphs
# from modules.transforms.liftings.graph2cell.discrete_configuration_complex_lifting import (
#     DiscreteConfigurationComplexLifting,
# )


# class TestDiscreteConfigurationComplexLifting:
#     """Test the DiscreteConfigurationComplexLifting class."""

#     def setup_method(self):
#         # Load the graph
#         self.dataset = load_simple_configuration_graphs()

#         # Initialise the DiscreteConfigurationComplexLifting class
#         self.liftings = [
#             DiscreteConfigurationComplexLifting(
#                 k=2, preserve_edge_attr=True, feature_aggregation="mean"
#             ),
#             DiscreteConfigurationComplexLifting(
#                 k=2, preserve_edge_attr=True, feature_aggregation="sum"
#             ),
#             DiscreteConfigurationComplexLifting(
#                 k=2, preserve_edge_attr=True, feature_aggregation="concat"
#             ),
#         ]

#     def test_lift_topology(self):
#         # Test the lift_topology method

#         expected_incidences_data_0 = (
#             torch.tensor(
#                 [
#                     [
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                     ],
#                 ]
#             ),
#             torch.tensor([]),
#         )

#         expected_incidences_data_1 = (
#             torch.tensor(
#                 [
#                     [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
#                     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
#                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#                 ]
#             ),
#             torch.tensor([]),
#         )

#         expected_incidences_data_2 = (
#             torch.tensor(
#                 [
#                     [
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         1.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                     ],
#                 ]
#             ),
#             torch.tensor(
#                 [
#                     [0.0, 0.0],
#                     [0.0, 0.0],
#                     [1.0, 0.0],
#                     [1.0, 0.0],
#                     [1.0, 0.0],
#                     [0.0, 0.0],
#                     [0.0, 0.0],
#                     [0.0, 0.0],
#                     [1.0, 0.0],
#                     [0.0, 0.0],
#                     [0.0, 1.0],
#                     [0.0, 1.0],
#                     [0.0, 1.0],
#                     [0.0, 1.0],
#                     [0.0, 0.0],
#                     [0.0, 0.0],
#                 ]
#             ),
#         )

#         for lifting in self.liftings:
#             lifted_data = lifting.forward(self.dataset[0].clone())
#             assert (
#                 expected_incidences_data_0[0] == lifted_data.incidence_1.to_dense()
#             ).all(), f"Something is wrong with incidence_1 for graph 0, {lifting.feature_aggregation} aggregation."

#             assert (
#                 expected_incidences_data_0[1] == lifted_data.incidence_2.to_dense()
#             ).all(), f"Something is wrong with incidence_2 for graph 0, {lifting.feature_aggregation} aggregation."

#             lifted_data = lifting.forward(self.dataset[1].clone())
#             assert (
#                 expected_incidences_data_1[0] == lifted_data.incidence_1.to_dense()
#             ).all(), f"Something is wrong with incidence_1 for graph 1, {lifting.feature_aggregation} aggregation."

#             assert (
#                 expected_incidences_data_1[1] == lifted_data.incidence_2.to_dense()
#             ).all(), f"Something is wrong with incidence_2 for graph 1, {lifting.feature_aggregation} aggregation."

#             lifted_data = lifting.forward(self.dataset[2].clone())
#             assert (
#                 expected_incidences_data_2[0] == lifted_data.incidence_1.to_dense()
#             ).all(), f"Something is wrong with incidence_1 for graph 2, {lifting.feature_aggregation} aggregation."

#             assert (
#                 expected_incidences_data_2[1] == lifted_data.incidence_2.to_dense()
#             ).all(), f"Something is wrong with incidence_2 for graph 2, {lifting.feature_aggregation} aggregation."
