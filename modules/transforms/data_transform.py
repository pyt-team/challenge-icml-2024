import torch_geometric

from modules.transforms.data_manipulations.manipulations import (
    IdentityTransform,
    KeepOnlyConnectedComponent,
    NodeDegrees,
    NodeFeaturesToFloat,
    OneHotDegreeFeatures,
)
from modules.transforms.feature_liftings.feature_liftings import ProjectionSum
from modules.transforms.liftings.graph2cell.cycle_lifting import (
    CellCycleLifting,
)
from modules.transforms.liftings.graph2combinatorial.curve_lifting import (
    CurveLifting,
)
from modules.transforms.liftings.graph2combinatorial.ring_close_atoms_lifting import (
    CombinatorialRingCloseAtomsLifting,
)
from modules.transforms.liftings.graph2combinatorial.sp_lifting import (
    SimplicialPathsLifting,
)
from modules.transforms.liftings.graph2hypergraph.expander_graph_lifting import (
    ExpanderGraphLifting,
)
from modules.transforms.liftings.graph2hypergraph.forman_ricci_curvature_lifting import (
    HypergraphFormanRicciCurvatureLifting,
)
from modules.transforms.liftings.graph2hypergraph.kernel_lifting import (
    HypergraphKernelLifting,
)
from modules.transforms.liftings.graph2hypergraph.knn_lifting import (
    HypergraphKNNLifting,
)
from modules.transforms.liftings.graph2hypergraph.mapper_lifting import (
    MapperLifting,
)
from modules.transforms.liftings.graph2hypergraph.modularity_maximization_lifting import (
    ModularityMaximizationLifting,
)
from modules.transforms.liftings.graph2simplicial.clique_lifting import (
    SimplicialCliqueLifting,
)
from modules.transforms.liftings.graph2simplicial.eccentricity_lifting import (
    SimplicialEccentricityLifting,
)
from modules.transforms.liftings.graph2simplicial.graph_induced_lifting import (
    SimplicialGraphInducedLifting,
)
from modules.transforms.liftings.graph2simplicial.latentclique_lifting import (
    LatentCliqueLifting,
)
from modules.transforms.liftings.graph2simplicial.line_lifting import (
    SimplicialLineLifting,
)
from modules.transforms.liftings.graph2simplicial.vietoris_rips_lifting import (
    SimplicialVietorisRipsLifting,
)
from modules.transforms.liftings.hypergraph2combinatorial.universal_strict_lifting import (
    UniversalStrictLifting,
)
from modules.transforms.liftings.hypergraph2simplicial.heat_lifting import (
    HypergraphHeatLifting,
)
from modules.transforms.liftings.pointcloud2hypergraph.mogmst_lifting import (
    MoGMSTLifting,
)
from modules.transforms.liftings.pointcloud2hypergraph.pointnet_lifting import (
    PointNetLifting,
)
from modules.transforms.liftings.pointcloud2hypergraph.voronoi_lifting import (
    VoronoiLifting,
)
from modules.transforms.liftings.pointcloud2simplicial.alpha_complex_lifting import (
    AlphaComplexLifting,
)
from modules.transforms.liftings.pointcloud2simplicial.delaunay_lifting import (
    DelaunayLifting,
)

TRANSFORMS = {
    # Graph -> Hypergraph
    "HypergraphKNNLifting": HypergraphKNNLifting,
    "HypergraphKernelLifting": HypergraphKernelLifting,
    "ExpanderGraphLifting": ExpanderGraphLifting,
    "HypergraphFormanRicciCurvatureLifting": HypergraphFormanRicciCurvatureLifting,
    "MapperLifting": MapperLifting,
    "ModularityMaximizationLifting": ModularityMaximizationLifting,
    # Graph -> Simplicial Complex
    "SimplicialCliqueLifting": SimplicialCliqueLifting,
    "SimplicialEccentricityLifting": SimplicialEccentricityLifting,
    "SimplicialGraphInducedLifting": SimplicialGraphInducedLifting,
    "SimplicialLineLifting": SimplicialLineLifting,
    "SimplicialVietorisRipsLifting": SimplicialVietorisRipsLifting,
    "LatentCliqueLifting": LatentCliqueLifting,
    # Graph -> Cell Complex
    "CellCycleLifting": CellCycleLifting,
    # Graph -> Combinatorial Complex
    "CombinatorialRingCloseAtomsLifting": CombinatorialRingCloseAtomsLifting,
    "CurveLifting": CurveLifting,
    "SimplicialPathsLifting": SimplicialPathsLifting,
    # Point Cloud -> Simplicial Complex,
    "AlphaComplexLifting": AlphaComplexLifting,
    # Point-cloud -> Simplicial Complex
    "DelaunayLifting": DelaunayLifting,
    # Pointcloud -> Hypergraph
    "VoronoiLifting": VoronoiLifting,
    "MoGMSTLifting": MoGMSTLifting,
    "PointNetLifting": PointNetLifting,
    # Hypergraph -> Combinatorial Complex
    "UniversalStrictLifting": UniversalStrictLifting,
    # Hypergraph -> Simplicial Complex
    "HypergraphHeatLifting": HypergraphHeatLifting,
    # Feature Liftings
    "ProjectionSum": ProjectionSum,
    # Data Manipulations
    "Identity": IdentityTransform,
    "NodeDegrees": NodeDegrees,
    "OneHotDegreeFeatures": OneHotDegreeFeatures,
    "NodeFeaturesToFloat": NodeFeaturesToFloat,
    "KeepOnlyConnectedComponent": KeepOnlyConnectedComponent,
}


class DataTransform(torch_geometric.transforms.BaseTransform):
    """Abstract class that provides an interface to define a custom data lifting.

    Parameters
    ----------
    transform_name : str
        The name of the transform to be used.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, transform_name, **kwargs):
        super().__init__()

        kwargs["transform_name"] = transform_name
        self.parameters = kwargs

        self.transform = (
            TRANSFORMS[transform_name](**kwargs)
            if transform_name is not None
            else None
        )

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        """Forward pass of the lifting.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        transformed_data : torch_geometric.data.Data
            The lifted data.
        """
        return self.transform(data)


if __name__ == "__main__":
    _ = DataTransform()
