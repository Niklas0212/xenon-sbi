# Utils

This folder contains shared utility code used across training, evaluation, and notebook analysis.

It is the main place for reusable building blocks (model classes, preprocessing, training loops, posterior visualization).

## File Overview

### [utils/architectures.py](utils/architectures.py)

Defines neural network architectures for SBI classifiers.

- Core class: `BaseMLP` (concatenates `x` and `theta`, outputs match probability)
- Thin named wrappers for feature variants:
	- WimPy feature models (`Full_MLP`, `Hist_MLP`, `Ntot_MLP`, `Highest_MLP`, ...)
	- S1-S2 models (`S1S2_signal`, `S1S2_signal_bg`, `HistS1S2_MLP`)

These class names are referenced by model config entries in [configs/config.py](configs/config.py).

### [utils/processing.py](utils/processing.py)

Data handling and preprocessing helpers.

- `preprocess_features`: model-aware feature slicing and transforms
	- Applies log transform for WimPy feature vectors
	- Keeps S1-S2 histograms in their expected format
- `make_negative_pairs`: creates positive/negative `(x, theta)` pairs for classifier-based SBI
- `make_dataloaders`: reproducible train/val/test splits
- `get_matching_pairs`, `load_matching_pairs`: retrieve only true matching pairs for diagnostics/metrics
- `filter_by_parameter_ranges`: optional parameter-window filtering

This module is the data interface used by both training and performance code.

### [utils/training.py](utils/training.py)

Training and checkpoint utilities.

- `train`: full training loop with
	- BCE loss
	- validation tracking
	- early stopping
	- best-model restore
	- checkpoint save
	- optional test tracking and Optuna pruning support
- `plot_training_summary`: plots loss/accuracy curves from saved checkpoints

Checkpoint dictionaries from this module are consumed by notebooks and performance scripts.

### [utils/posteriors.py](utils/posteriors.py)

Posterior computation and visualization utilities.

Main capabilities include:

- Posterior grid construction from classifier scores (`posterior_grid`)
- HPD contour computation and plotting
- Representation conversions (`cp <-> sigma_p`) and axis formatting
- Null-spectrum exclusions and official limit overlays
- WimPy and S1-S2 plotting workflows (single posterior, grids, comparison helpers)
- Background-realization and limit-statistics helpers used in advanced S1-S2 notebooks

This is the largest utility module and is mainly used from notebooks.

## Typical Workflow Across Utilities

1. Select architecture via config/model name ([utils/architectures.py](utils/architectures.py)).
2. Load and preprocess data, then build SBI training pairs ([utils/processing.py](utils/processing.py)).
3. Train and save best checkpoint ([utils/training.py](utils/training.py)).
4. Visualize posteriors/exclusions from checkpointed models ([utils/posteriors.py](utils/posteriors.py)).

## Notes

- Utility functions are intentionally shared by multiple folders (`sbi_training`, `performances`, `sbi_notebooks`).
- When changing function signatures here, check downstream callers in:
	- [sbi_training](sbi_training)
	- [performances](performances)
	- [sbi_notebooks](sbi_notebooks)
