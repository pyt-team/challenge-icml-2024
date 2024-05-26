# DL 2 Project - Reproduction of E(n) Equivariant Simplicial Message Passing Network

This project is meant to showcase the effort in replication the paper of Eijkelboom et al about message passing along simplices. Our aim was to re-implement the proposed network architecture in the up and comming TopoX package suite. This suite is a newly developed set of packages that facilitate the pipeline of Topological Deep Learning, from lifting techniques to actual models of message passing.

## Setup
To setup from a clean environment we will reproduce the steps as mentioned in the `challenge-icml-2024` and an additional one.

1. Create the conda environment to be used
   ```bash
   conda create -n topox python=3.11.3
   conda activate topox
   ```
2. Install the required packages and the TopoX suite including the code reqquiremed from the `challenge-icml-2024` repo
   ```bash
   pip install -e '.[all]'
   ```
3. Install pytorch related libraries

      ```bash
      pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu115
      pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu115.html
      pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu115.html
      ```
4. Wandb for saving model information
    ```bash
    pip install wandb
    ```
## Experiments

There are two important directions the experiments follow: execution efficiency, performance of the network.
First, we will evaluate the efficiency of the lifting procedure for different complexes (Alpha Complex, Vietoris-Rips), the efficiency of pre-computing the invariance relationships prior to the forward pass and the actual
performance in the QM9 dataset.

To execute an experiment activate the `topox` environment created above and go into `src`. The file `main.py` will
run the experiment, some optional paramters are explained below.

+ `--lift_type <lift>`: the type of lifting procedure `alpha` or `rips`
+ `--batch_size <size>` the size of each batch
+ `--target_name <name>`: the target molecular property to train/predict for
+ `--debug `: to run a smaller subset of the dataset for testing purpouses
+ `--pre_proc`: wether the invariances should be precomputed during the lift procedure or not (beware it's vary time consuming)

