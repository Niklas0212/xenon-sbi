# xenon-sbi

Simulation-based inference (SBI) pipeline for dark matter direct-detection studies, with two main representations:

- WimPy recoil-spectrum features
- Xenon S1-S2 histogram features (signal-only and signal+background)

The repository covers data generation, model training, posterior visualization, and quantitative performance evaluation (calibration, distances, divergence, and halo sensitivity).

## Project Pipeline

1. Generate or prepare datasets in [data](data).
2. Train classifier-based SBI models with [sbi_training](sbi_training).
3. Analyze posteriors and exclusion behavior in [sbi_notebooks](sbi_notebooks).
4. Run benchmark metrics and halo-sensitivity checks in [performances](performances).

## Top-Level Structure

- [configs](configs): central configuration and model registry.
- [data](data): simulation/generation scripts and dataset products.
- [sbi_training](sbi_training): training entry points for WimPy and S1-S2 models.
- [sbi_notebooks](sbi_notebooks): notebook analysis for posterior behavior and cross-setup comparison.
- [performances](performances): metric computations and performance notebooks.
- [utils](utils): shared architectures, preprocessing, training loops, posterior plotting utilities.
- [analytical](analytical): analytical 1D Poisson-posterior tooling for the WimPy case.
- [hpo](hpo): Optuna scripts and study databases.
- [models](models): trained checkpoints (organized by experiment family).
- [WimPyDD](WimPyDD): external/packaged dark-matter physics machinery used by simulation workflows.
- [figures](figures): figure outputs and report/thesis plot targets.

## Most Important Files

- [environment.yml](environment.yml): environment specification.
- [configs/config.py](configs/config.py): model definitions, defaults, scheduler/optimizer wiring, load helpers.
- [sbi_training/training_wimpy.py](sbi_training/training_wimpy.py): train WimPy-feature SBI models.
- [sbi_training/training_s1s2.py](sbi_training/training_s1s2.py): train S1-S2 SBI models (signal-only or background-inclusive).
- [utils/processing.py](utils/processing.py): feature preprocessing and pair/dataloader construction.
- [utils/training.py](utils/training.py): training loop, early stopping, checkpointing, training-curve summaries.
- [utils/posteriors.py](utils/posteriors.py): posterior grids, HPD contours, exclusions, plotting helpers.
- [performances/performances.py](performances/performances.py): end-to-end performance metric evaluation CLI.
- [performances/metrics.py](performances/metrics.py): metric implementations and plotting helpers.

## Folder Guides

Detailed subfolder documentation is available in:

- [data/README.md](data/README.md)
- [sbi_training/README.md](sbi_training/README.md)
- [sbi_notebooks/README.md](sbi_notebooks/README.md)
- [performances/README.md](performances/README.md)
- [utils/README.md](utils/README.md)
- [analytical/README.md](analytical/README.md)
- [hpo/README.md](hpo/README.md)
- [models/README.md](models/README.md)

## Quick Start

### 1) Data generation

Use the data pipeline from [data/generate_dataset.py](data/generate_dataset.py) and related tools in [data](data).

Example:

```bash
python -m data.generate_dataset --n_train 300000 --datatag low --halo_option shm
```

### 2) Train models

WimPy example:

```bash
python -m sbi_training.training_wimpy --n_train 300000 --modelname ntothighest --datatag low --halos shm
```

S1-S2 example:

```bash
python -m sbi_training.training_s1s2 --halos shm --mu_bg 150 --bins 10 --batch_size 1024
```

### 3) Inspect posteriors

Start with:

- [sbi_notebooks/sbi_wimpy.ipynb](sbi_notebooks/sbi_wimpy.ipynb)
- [sbi_notebooks/sbi_s1s2_signal_only.ipynb](sbi_notebooks/sbi_s1s2_signal_only.ipynb)
- [sbi_notebooks/old approach - uncertainty band/sbi_s1s2_signal_bg_old.ipynb](sbi_notebooks/old%20approach%20-%20uncertainty%20band/sbi_s1s2_signal_bg_old.ipynb)
- [sbi_notebooks/new approach - posterior averaging/sbi_s1s2_signal_bg_new.ipynb](sbi_notebooks/new%20approach%20-%20posterior%20averaging/sbi_s1s2_signal_bg_new.ipynb)

### 4) Evaluate model quality

```bash
python -m performances.performances --n-total 5000 --datatags low mid high --halo default
```

## Notes for New Contributors

- Keep configuration changes centralized in [configs/config.py](configs/config.py).
- Utility signature changes in [utils](utils) usually affect training scripts, notebooks, and performance code simultaneously.
- Many paths follow strict naming conventions; preserve them unless intentionally migrating old checkpoints/datasets.

## Data Availability and Repository Layout Notes

### Reduced datasets on GitHub

Training was performed on **n = 300,000** spectra per dataset. The full datasets
are several gigabytes and exceed GitHub's file-size limits, so this repository
ships only a **reduced subset of the first n = 30,000 spectra** for each dataset,
sufficient to run and inspect the pipeline. The same applies to the ER background
pool, which is likewise truncated here.

The **trained model checkpoints** in [models](models), however, correspond to the
**full n = 300,000** training runs. Re-training on the reduced GitHub subset will
therefore not reproduce the published checkpoints; it is intended only for testing
the pipeline end-to-end. To reproduce the full results, regenerate the complete
datasets with the data-generation scripts (see Quick Start) using n = 300,000.

### Two exclusion approaches in `sbi_notebooks`

The [sbi_notebooks](sbi_notebooks) folder contains two subfolders reflecting the
two methods we used to derive the 90% upper-limit exclusions:

- **`old approach - uncertainty bands`**: the original approach, deriving the
  exclusion by marginalizing the posterior over background realizations and
  extracting per-realization quantile bands.
- **`new approach - posterior averaging`**: the alternative approach used for the
  final results, which averages the posteriors over many background realizations
  and applies a single 90% HPD exclusion to the averaged posterior.