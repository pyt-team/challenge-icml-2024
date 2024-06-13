# Molecule Ring-Based Lifting (Graph to Cell)

Based on [Jin et al. (2018)](https://arxiv.org/abs/1802.04364), this implementation aims to condense a molecule ring into a single element. Utilizing the [QM9 dataset](https://paperswithcode.com/dataset/qm9), a benchmark dataset for molecule prediction, we develop a method to treat rings as 2-cell structures by identifying them through the provided SMILES representations of each molecule.

Here, the elements are the following:
- **Nodes**: Atoms in the molecule.
- **Edges**: Bonds between atoms.
- **2-cells**: Rings in the molecule.

This pull request is done under the team formed by: Bertran Miquel Oliver, Manel Gil Sorribes, Alexis Molina