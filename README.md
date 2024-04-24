# challenge-icml-2024
Official repository for the ICML 2024 Topological Deep Learning Challenge, hosted by the Geometry-grounded Representation Learning and Generative Modeling (GRaM) Workshop.


## 🦾 Contributing to challenge-icml-2024
To develop on your machine, here are some tips.

First, we recommend using Python 3.11.3, which is the python version used to run the unit-tests.

For example, create a conda environment:
   ```bash
   conda create -n tmx python=3.11.3
   conda activate tmx
   ```

Then:

1. Clone a copy of tmx from source:

   ```bash
   git clone git@github.com:pyt-team/challenge-icml-2024.git
   cd challenge-icml-2024
   ```

2. Install tmx in editable mode:

   ```bash
   pip install -e '.[all]'
   ```
   **Notes:**
   - Requires pip >= 21.3. Refer: [PEP 660](https://peps.python.org/pep-0660/).
   - On Windows, use `pip install -e .[all]` instead (without quotes around `[all]`).

4. Install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.

      ```bash
      pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
      pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
      pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
      ```

      where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).

5. Ensure that you have a working tmx installation by running the entire test suite with

   ```bash
   pytest
   ```

    In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

6. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```
