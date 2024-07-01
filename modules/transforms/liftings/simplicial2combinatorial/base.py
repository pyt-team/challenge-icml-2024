from modules.transforms.liftings.lifting import SimplicialLifting


class Simplicial2CombinatorialLifting(SimplicialLifting):
    r"""Abstract class for lifting graphs to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optiona""l
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "simplicial2combinatorial"

