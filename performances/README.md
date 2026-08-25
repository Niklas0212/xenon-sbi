# Performance Evaluation

This folder contains evaluation tools for checking whether learned SBI posteriors are not only accurate, but also statistically reliable.

## Why performance metrics matter

For posterior inference, a single number like validation accuracy is not enough.

- A model can score well on training-style metrics while returning overconfident or miscalibrated posteriors.
- Scientific conclusions depend on uncertainty quantification, so calibration and posterior shape are as important as point accuracy.
- Different halo assumptions can induce subtle shifts or biases. Performance metrics make this visible and comparable.

In short: we need metrics that test calibration, agreement with analytical references, and geometric quality of posteriors.

## What is evaluated here

The core metrics are implemented in `performances/metrics.py`:

1. Coverage (HPD credible regions)
- Tests whether nominal credibility levels match empirical containment of true parameters.
- Summarized by signed and absolute calibration deviations.

2. Jensen-Shannon divergence (JSD)
- Compares learned posterior grids against analytical Poisson posterior grids.
- Measures distribution-level agreement (not only point estimates).

3. Euclidean and Mahalanobis distances
- Euclidean: point error in parameter space.
- Mahalanobis: error normalized by posterior covariance (uncertainty-aware error).
- Together they distinguish bias from over/under-confidence.

4. Posterior volume (in halo-comparison workflows)
- Used as a sharpness/precision indicator and often shown relative to a reference model.

## Where to start (recommended)

Start with the notebooks in `performances/notebooks/` to build intuition for how performance metrics are used and interpreted in practice.

Then use `performances/metrics.py` when you need a specific function.

- It is intentionally long and contains all core metric implementations plus many plotting/helper functions.
- It is best used as a function-level reference rather than a first-read file.

`performances/performances.py` is the script that runs the end-to-end performance evaluation.

## Notebooks: what each one does

- `performances/notebooks/metrics_coverages.ipynb`
	Coverage calibration checks on validation/test data.

- `performances/notebooks/metrics_eucl_maha.ipynb`
	Euclidean and Mahalanobis diagnostics, including interpretation under zero-event/exclusion-region effects.

- `performances/notebooks/performances_wimpy.ipynb`
	End-to-end model comparison for WimPy regimes (`low`, `mid`, `high`) using training and posterior-quality metrics.

- `performances/notebooks/performances_s1s2.ipynb`
	Readout/comparison of Xenon S1S2 checkpoint metrics (signal-only and signal+background configurations).

- `performances/notebooks/halo_coverages_wimpy_extreme.ipynb`
	Concept demonstration of halo misspecification effects using extreme halos (`x1`, `x2`) to amplify differences.

- `performances/notebooks/halo_coverages_wimpy_real.ipynb`
	Realistic halo sensitivity study (`default`, `shm`, `shmpp`, `lmc`) where expected differences are subtle.

- `performances/notebooks/halo_coverages_s1s2_extreme.ipynb`
	Halo-comparison workflow for the S1S2 representation.

## Halo-model sensitivity: how to interpret

We evaluate halo sensitivity by comparing posteriors from models trained under different halo assumptions against data generated from a chosen reference halo.

- Coverage changes indicate calibration drift under misspecification.
- Euclidean/Mahalanobis shifts indicate bias and uncertainty mismatch.
- Relative volume indicates whether posteriors become artificially sharp or broad under alternative halo assumptions.

Two complementary settings are used:

1. Extreme halos (`x1`, `x2`): conceptual stress test, large visible effects.
2. Realistic halos (`shm`, `shmpp`, `lmc`, `default`): physically relevant regime, typically smaller differences.

This split helps separate method sensitivity from physically realistic effect size.

## CLI usage and cached outputs

You can run full evaluations via:

```bash
python -m performances.performances --n-total 5000 --datatags low mid high --halo default
```

Results are cached in `performances/performances_results/` to avoid recomputation for repeated configurations.

