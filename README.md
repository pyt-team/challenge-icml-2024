# Molecule Ring & Close Atoms Lifting (Graph to Combinatorial)
This notebook imports QM9 dataset and applies a lifting from a graph molecular representation to a combinatorial complex. Then, a neural network is run using the loaded data.

Using [QM9 dataset](https://paperswithcode.com/dataset/qm9), we implement a lifting from a molecule graph to a combinatorial complex based on two points:
- The ring information of the molecule. Rings will be represented as 2-cells in the combinatorial complex.
- The distance between atoms in the molecule. Distances between atoms will be computed. If the atoms are under a predefined threshold, they will be considered as close and groupped together. This clusters will be introduced as hyperedges in the combinatorial complex.

So far, to the best of our knowledge, it is the first implementation of a molecule as a combinatorial complex, combining both hypergraphs and cell complexes.

Here, the elements are the following:
- **Nodes**: Atoms in the molecule.
- **Edges**: Bonds between atoms.
- **Hyperedges**: Clusters of atoms that are close to each other.
- **2-cells**: Rings in the molecule.

This pull request is done under the team formed by: Bertran Miquel Oliver, Manel Gil Sorribes, Alexis Molina
