from topomodelx.utils.sparse import from_sparse
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from torch_geometric.data import Data

from modules.transforms.liftings.simplicial2combinatorial.base import (
    Simplicial2CombinatorialLifting,
)


class CofaceCCLifting(Simplicial2CombinatorialLifting):
    def __init__(self, keep_features=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_features = keep_features

    def lift_topology(self, data: Data) -> dict:

        cells = []
        ranks = []

        # Initialize the CC given a torch_geometric.Data
        # graph
        ## Add 0-cells
        for cell in range(data["x_0"].size(0)):
            cells.append(cell)
            ranks.append(0)

        ## Add 1-cells
        for inc_c_1 in data["incidence_1"].to_dense().T:
            # Get the 0-cells that are incident to the 1-cell
            cell_0_bound = inc_c_1.nonzero().flatten().tolist()
            assert(len(cell_0_bound) == 2)
            cells.append((cell_0_bound[0], cell_0_bound[1]))
            ranks.append(1)

        ## Add 2-cells
        for inc_c_2 in data["incidence_2"].to_dense().T:
            # Get the 1-cells that are incident to the 2-cell
            cell_1_bound = inc_c_2.nonzero().flatten()
            # Get the 0-cells that are incident to the 1-cells
            cell_0_bound = data["incidence_1"].to_dense().T[cell_1_bound].nonzero()
            # Get the actual 0-cells since nonzero()
            # indexes in 2D
            cell_0_bound = cell_0_bound[:, 1]
            # Remove redudants and convert to tuple
            cell_0 = tuple(set(cell_0_bound.tolist()))
            cells.append(cell_0)
            ranks.append(2)

        cc = CombinatorialComplex(cells, ranks=ranks, graph_based=False)

        # Iterate over the 2-cells and add the 3-cells
        for cell in cc.skeleton(rank=2):
            # Get the coface of the 2-cell
            indices, coface = cc.coadjacency_matrix(2, 1, index=True)

            # Get the indices of the 2-cell that are co-adjacent
            coface_indices = coface.todense()[indices[cell]].nonzero()[1].tolist()
            cell_3 = set(cell)

            # Iterate over the indices of the 2-cells
            # and add their 0-cells as a 3-cell
            for idx in coface_indices:
                cell_3 = cell_3.union(set(cc.skeleton(rank=2)[idx]))

            # Adding a rank 3 cell with less than 4 vertices
            # will take this cell from the skeleton of 2-cells if it exists
            # so in the interest of keeping features the user
            # can choose to recompute all feature embeddings
            if len(cell_3) < 4 and self.keep_features:
                continue
            # Get the cofaces incident to the 2-cell `cell` and add `cell` to the set
            cc.add_cell(cell_3, rank=3)

        # If the user wants to keep the features
        # from the r-cells aside from the first x_0
        if self.keep_features:
            new_graph = {"x_0": data["x_0"], "x_1": data["x_1"], "x_2": data["x_2"]}
        else:
            new_graph = {"x_0": data["x_0"]}

        # Create the dirac operator matrix
        #new_graph['dirac'] = cc.dirac_operator_matrix()

        # Create the incidence, adjacency and laplacian matrices
        for r in range(cc.dim+1):
            if r > 0:
                new_graph[f"laplacian_{r}"] = cc.laplacian_matrix(r)
            if r < cc.dim:
                new_graph[f"incidence_{r+1}"] = from_sparse(cc.incidence_matrix(r , r+1, incidence_type="up"))
            new_graph[f"adjacency_{r}"] = from_sparse(cc.adjacency_matrix(r, r+1))

        return new_graph

    def forward(self, data: Data) -> Data:
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        # Make sure to remove passing of duplicated data
        # so that the constructor of Data does not raise an error
        for i in range(4):
            if f"x_{i}" in initial_data:
                del initial_data[f"x_{i}"]
            if f"incidence_{i}" in initial_data:
                del initial_data[f"incidence_{i}"]
            if f"adjacency_{i}" in initial_data:
                del initial_data[f"adjacency_{i}"]
        return Data(**initial_data, **lifted_topology)
