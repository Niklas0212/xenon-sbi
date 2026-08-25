# SBI Notebooks

This folder contains analysis notebooks for inspecting trained SBI models and visualizing posterior inference behavior across different data representations.

The notebooks are not training entry points. Training is done in the scripts under [sbi_training](../sbi_training).

## What You Find Here

- Training diagnostics from saved checkpoints (loss and accuracy history)
- Posterior examples for selected parameter points
- Posterior grids across parameter space
- Exclusion-style summaries from HPD posteriors
- In the signal+background case: background-realization effects and marginalized limit bands

## Notebook Guide

### [sbi_wimpy.ipynb](sbi_wimpy.ipynb)

Use this for the WimPy (energy-spectrum) representation.

- Loads a trained WimPy model and validation data
- Plots training curves
- Shows single and paired posterior examples
- Builds a grid of posteriors across parameter space
- Computes HPD-based exclusion from a null spectrum and can compare to official XenonNT limits

This is the cleanest starting point if you want a baseline SBI inference walkthrough.

### [sbi_s1s2_signal_only.ipynb](sbi_s1s2_signal_only.ipynb)

Use this for S1S2 histograms in the signal-only setup.

- Mirrors the same core workflow as the WimPy notebook
- Focuses on S1S2 model checkpoints and S1S2 validation loader
- Produces posterior examples, posterior grids, and HPD exclusions without background contamination

Use this to isolate representation effects from background effects.

### [sbi_s1s2_signal_bg.ipynb](sbi_s1s2_signal_bg.ipynb)

Use this for S1S2 with explicit signal+background spectra (realistic contamination).

- Starts from the signal-only workflow but with nonzero background rate
- Demonstrates how posterior localization and width change with background fluctuations
- Shows why a single HPD contour is not stable in this regime
- Adds marginalized limit statistics and confidence-style bands
- Includes tools such as critical-mass selection and multi-realization limit visualization

This is the main notebook for robust limits in the presence of background noise.

### [sbi_comparison.ipynb](sbi_comparison.ipynb)

Use this to compare inference outputs across representations and assumptions.

- Compares three settings side by side:
- Compares WimPy against S1S2 signal-only and S1S2 signal+background in one layout
- Uses shared target points to visualize posterior differences in a unified layout
- Compares exclusion behavior between setups

This is best for summary figures and high-level model comparison.

## Suggested Reading Order

1. [sbi_wimpy.ipynb](sbi_wimpy.ipynb)
2. [sbi_s1s2_signal_only.ipynb](sbi_s1s2_signal_only.ipynb)
3. [sbi_s1s2_signal_bg.ipynb](sbi_s1s2_signal_bg.ipynb)
4. [sbi_comparison.ipynb](sbi_comparison.ipynb)

This order gives a progression from baseline inference, to S1S2 translation, to realistic background effects, and finally cross-setup comparison.
