import torch_geometric

from modules.transforms.data_manipulations.manipulations import (
    IdentityTransform,
    KeepOnlyConnectedComponent,
    NodeDegrees,
    NodeFeaturesToFloat,
    OneHotDegreeFeatures,
    FilterEnoughSimplices,
    InputPreproc,
    LabelPreproc
)
from modules.transforms.feature_liftings.feature_liftings import ProjectionSum, ProjectionMean
from modules.transforms.liftings.graph2cell.cycle_lifting import CellCycleLifting
from modules.transforms.liftings.graph2hypergraph.knn_lifting import (
    HypergraphKNNLifting,
)
from modules.transforms.liftings.graph2simplicial.clique_lifting import (
    SimplicialCliqueLifting,
)
from modules.transforms.liftings.graph2simplicial.vietoris_rips_lifting import (
    SimplicialVietorisRipsLifting,
    InvariantSimplicialVietorisRipsLifting 
)

from modules.transforms.liftings.graph2simplicial.alpha_complex_lifting import (
    SimplicialAlphaComplexLifting
)

TRANSFORMS = {
    # Graph -> Hypergraph
    "HypergraphKNNLifting": HypergraphKNNLifting,
    # Graph -> Simplicial Complex
    "SimplicialCliqueLifting": SimplicialCliqueLifting,
    "SimplicialVietorisRipsLifting": SimplicialVietorisRipsLifting,
    "InvariantSimplicialVietorisRipsLifting": InvariantSimplicialVietorisRipsLifting,
    "SimplicialAlphaComplexLifting": SimplicialAlphaComplexLifting,
    # Graph -> Cell Complex
    "CellCycleLifting": CellCycleLifting,
    # Feature Liftings
    "ProjectionSum": ProjectionSum,
    "ProjectionMean": ProjectionMean,
    # Data Manipulations
    "Identity": IdentityTransform,
    "NodeDegrees": NodeDegrees,
    "OneHotDegreeFeatures": OneHotDegreeFeatures,
    "NodeFeaturesToFloat": NodeFeaturesToFloat,
    "KeepOnlyConnectedComponent": KeepOnlyConnectedComponent,
    "FilterEnoughSimplices": FilterEnoughSimplices,
    "InputPreproc": InputPreproc,
    "LabelPreproc": LabelPreproc,
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
            TRANSFORMS[transform_name](**kwargs) if transform_name is not None else None
        )

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
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
