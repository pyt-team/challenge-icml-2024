# Molecule Ring-Based Lifting (Graph to Cell)

Based on [Jin et al. (2018)](https://arxiv.org/abs/1802.04364), this implementation aims to condense a molecule ring into a single element. Utilizing the [QM9 dataset](https://paperswithcode.com/dataset/qm9), a benchmark dataset for molecule prediction, we develop a method to treat rings as 2-cell structures by identifying them through the provided SMILES representations of each molecule.

Here, the elements are the following:
- **Nodes**: Atoms in the molecule.
- **Edges**: Bonds between atoms.
- **2-cells**: Rings in the molecule.

This pull request is done under the team formed by: Bertran Miquel Oliver, Manel Gil, Alexis Molina

## Questions

Feel free to contact us through GitHub issues on this repository, or through the [Geometry and Topology in Machine Learning slack](https://tda-in-ml.slack.com/join/shared_invite/enQtOTIyMTIyNTYxMTM2LTA2YmQyZjVjNjgxZWYzMDUyODY5MjlhMGE3ZTI1MzE4NjI2OTY0MmUyMmQ3NGE0MTNmMzNiMTViMjM2MzE4OTc#/). Alternatively, you can contact us via mail at any of these accounts: guillermo.bernardez@upc.edu, lev.telyatnikov@uniroma1.it.
