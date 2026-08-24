# Analytical posterior calculations

This folder is only relevant for the 1D WIMPyDD analytical case in this project.
It computes a grid-based Poisson posterior in the \(m_\chi, c_p\) plane and is
not the main workflow for the broader Xenon SBI pipeline.

## What it contains

- `lambda_calculation.py`: generate and save a lambda grid for a chosen region.
- `ppg_class.py`: main `PoissonPosteriorGrid` implementation.
- `lambda/`: stored grid files in `.npz` format.
- `notebooks/`: analytical notebook explorations.

## Usage

```bash
python -m analytical.lambda_calculation --datatag low --gridbins 100
```

Available datatags: `low`, `mid`, `high`.

This creates files named like:

```text
analytical/lambda/lambda_low_bins100.npz
```

The saved grid contains:

- `lambda_grid`
- `logm_vals`
- `logcp_vals`

The code reuses an existing file automatically if it is already present.

## Python example

```python
from analytical.ppg_class import PoissonPosteriorGrid
from data.mc_generator import mc_generator

obj = PoissonPosteriorGrid.load_or_compute(
    datatag="low",
    gridbins=100,
    mc_generator=mc_generator,
)

posterior = obj.posterior_nevents(n_observed=10)
```

This returns a normalized posterior density over the sampled parameter grid.
