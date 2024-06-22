from modules.transforms.liftings.lifting import PointCloudLifting


class PointCloud2SimplicialLifting(PointCloudLifting):
    r"""Abstract class for lifting pointclouds to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "pointcloud2simplicial"
