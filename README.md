# Molecule Ring & Functional Lifting (Graph to Combinatorial)
This notebook imports QM9 dataset and applies a lifting from a graph molecular representation to a combinatorial complex. Then, a neural network is run using the loaded data.

Using [QM9 dataset](https://paperswithcode.com/dataset/qm9), we implement a lifting from a molecule graph to a combinatorial complex based on two points:
- The ring information of the molecule. Rings will be represented as 2-cells in the combinatorial complex.
- Its functional groups will be add as hyperedges in the complex, i.e., 1-cells that sometimes connect more than two 0-cells. The functional groups are found by the SMARTS patterns, predifenied patterns that are used to identify functional groups in molecules.

So far, to the best of our knowledge, it is the first implementation of a molecule as a combinatorial complex, combining both hypergraphs and cell complexes.

Here, the elements are the following:
- **Nodes**: Atoms in the molecule.
- **Edges**: Bonds between atoms.
- **Hyperedges**: Clusters of atoms that are close to each other.
- **2-cells**: Rings in the molecule.

This pull request is done under the team formed by: Bertran Miquel Oliver, Manel Gil Sorribes, Alexis Molina