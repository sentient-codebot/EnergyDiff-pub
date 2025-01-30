# EnergyDiff

Diffusion-based energy time series data generation.

# Installation

Install the required packages. 

```bash
pip install -e .
```

# Run

1. Prepare the datasets by downloading from their original sources and running the preprocessing scripts (under `scripts/python/preprocessing`). 

2. Training, sampling, and evaluation. Refer to the scripts (both for bash, for slurm, and for python) under `scripts/` directory. 

# Licence
This repository adopts an MIT License. The conditional copula model is implemented by [conditonal-copula
](https://github.com/MauricioSalazare/conditonal-copula). 
