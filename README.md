# QUALIFIER 2023 - FPTC Evaluation

This repository contains the replication package of the experiments described in the paper _Towards a Prediction of Machine Learning Training Time to Support Continuous Learning Systems Development_.

## Requirements

- Install Miniconda (<https://docs.conda.io/en/latest/miniconda.html>)
- Create the conda environment using these commands from the root of the repository:
  - `conda env create -f environment.yml`
  - `conda activate fptc_eval`

## Structure

The repository is structured as follows:

- [`slope_variation`](./slope_variation/README.md): contains the codes to replicate the experiment conducted to answer RQ1.
- [`training_time`](./training_time/README.md): contains the codes to replicate the experiment conducted to answer RQ2.

Refer to the `README.md` file in each folder for more details on how to replicate the answers to the research questions.
