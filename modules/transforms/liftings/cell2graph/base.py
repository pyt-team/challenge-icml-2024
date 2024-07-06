from modules.transforms.liftings.lifting import CellComplexLifting


class Cell2GraphLifting(CellComplexLifting):
    r"""Abstract class for lifting cell complexes to graphs.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "cell2graph"
