# SBI Training

This folder contains the training entry points for the SBI classifiers used in this project.

- [sbi_training/training_wimpy.py](sbi_training/training_wimpy.py): trains models on WimPy recoil-spectrum features.
- [sbi_training/training_s1s2.py](sbi_training/training_s1s2.py): trains models on S1-S2 histogram features (signal-only or signal+background).

If you want inference/visualization, use the notebooks in [sbi_notebooks](sbi_notebooks) after training.

## Prerequisites

Before running training, make sure datasets exist:

- WimPy datasets under [data/datasets/wimpy](data/datasets/wimpy)
- S1-S2 signal datasets under [data/datasets/xenon/s1s2/pt](data/datasets/xenon/s1s2/pt)
- S1-S2 ER background CSV at [data/datasets/xenon/s1s2/ers/s1s2_ers.csv](data/datasets/xenon/s1s2/ers/s1s2_ers.csv)

Relevant defaults, model architectures, and optimizer/scheduler settings are defined in [configs/config.py](configs/config.py).

## Script 1: WimPy Training

### File

[sbi_training/training_wimpy.py](sbi_training/training_wimpy.py)

### What it does

1. Loads one or multiple halo-specific WimPy datasets.
2. Concatenates halos if multiple are requested.
3. Preprocesses features.
4. Builds positive/negative training pairs.
5. Creates train/val/test dataloaders.
6. Trains the selected classifier and saves a checkpoint.

### Main CLI arguments

- `--n_train` (default `300000`): samples per halo dataset.
- `--modelname` (default `full`): key from model config.
- `--datatag` (default `low`, choices `low|mid|high`): mass/coupling range split.
- `--halos` (default `default`): one or more halos, for example `default shm shmpp lmc`.
- `--batch_size`: train/eval batch size override.

### Examples

```bash
python -m sbi_training.training_wimpy --n_train 300000 --modelname ntothighest --datatag low --halos shm --batch_size 2048
python -m sbi_training.training_wimpy --n_train 300000 --modelname full --datatag low --halos shm shmpp --batch_size 4096
```

### Checkpoint output convention

- Single halo:
	- `models/wimpy/<halo>/<modelname>_n<n_train>_<datatag>_<halo>.pt`
- Multi-halo combined:
	- `models/wimpy/combined/<halo_tag>/<modelname>_n<n_train>_<datatag>_<halo_tag>.pt`

If the target checkpoint already exists, training is skipped.

## Script 2: S1-S2 Training

### File

[sbi_training/training_s1s2.py](sbi_training/training_s1s2.py)

### What it does

1. Loads and merges halo-specific S1-S2 signal event libraries.
2. Cleans NaN events.
3. Splits signal events into train/test (80/20).
4. Builds `SignalPlusBackgroundDataset` with sampled backgrounds.
5. Generates fixed histograms and positive/negative pairs.
6. Trains a classifier and saves a checkpoint.

Model selection is automatic:

- `mu_bg = 0` -> `S1S2_signal`
- `mu_bg > 0` -> `S1S2_signal_bg`

### Main CLI arguments

- `--n_train` (default `300000`): samples per halo dataset.
- `--halos` (default `shm`): one or more halos.
- `--mu_bg` (default `150`): expected ER background events (`0` for signal-only).
- `--bins` (default from config): S1 and S2 bins per axis.
- `--epochs`, `--batch_size`, `--patience`: training controls.

### Examples

```bash
python -m sbi_training.training_s1s2 --halos shm --mu_bg 0 --bins 10 --batch_size 1024
python -m sbi_training.training_s1s2 --halos shm shmpp --mu_bg 150 --bins 10 --batch_size 1024
```

### Checkpoint output convention

- Signal-only (`mu_bg=0`):
	- Single halo:
		- `models/xenon/signal_only/<halo>/<modelname>_bins<bins>_n<n_train>_<halo>.pt`
	- Multi-halo combined:
		- `models/xenon/signal_only/combined/<halo_tag>/<modelname>_bins<bins>_n<n_train>_<halo_tag>.pt`

- Signal+background (`mu_bg>0`, for example `150`):
	- Single halo:
		- `models/xenon/signal_bg_mu150/<halo>/<modelname>_bins<bins>_n<n_train>_<halo>.pt`
	- Multi-halo combined:
		- `models/xenon/signal_bg_mu150/combined/<halo_tag>/<modelname>_bins<bins>_n<n_train>_<halo_tag>.pt`

If the target checkpoint already exists, training is skipped.

## How This Connects to Notebooks

Training outputs are consumed by:

- [sbi_notebooks/sbi_wimpy.ipynb](sbi_notebooks/sbi_wimpy.ipynb)
- [sbi_notebooks/sbi_s1s2_signal_only.ipynb](sbi_notebooks/sbi_s1s2_signal_only.ipynb)
- [sbi_notebooks/sbi_s1s2_signal_bg.ipynb](sbi_notebooks/sbi_s1s2_signal_bg.ipynb)
- [sbi_notebooks/sbi_comparison.ipynb](sbi_notebooks/sbi_comparison.ipynb)
