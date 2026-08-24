# FUSE conversion pipeline

This folder contains the intermediate conversion workflow used to turn recoil
spectra into detector-level observables. The basic idea is simple: take a set of
nuclear recoil inputs generated earlier in the project, split them into chunks,
run the FUSE detector simulation on each chunk, and save the resulting `cs1` and
`cs2` signals.

This is a practical, chunked workflow rather than a polished pipeline. It is
functional, parallelized, and intentionally lightweight, but it is not optimized
for speed or elegance.

## What the scripts do

- `run_chunk.py`: processes one chunk of recoil events. It builds a FUSE input
  table from the recoil energies, runs the detector simulation, matches the
  returned microphysics and event outputs, and writes a CSV with the resulting
  `cs1`, `cs2`, and related bookkeeping columns.
- `py_submit_chunks.py`: checks which chunk outputs are missing and submits the
  corresponding Slurm jobs.
- `prepare_chunks.ipynb`: notebook used to prepare chunked input files for the
  FUSE runs.
- `merge_results.ipynb`: notebook used to combine chunk outputs into a single
  dataset.

## ER background workflow

The same idea is used for electronic recoil (ER) background generation:

- `run_er_chunk.py`: simulates one ER chunk and saves the resulting `cs1`/`cs2`
  values.
- `py_submit_er_chunks.py`: submits the ER jobs in parallel via Slurm.

This produces a background pool of ER events that can later be used for
training, comparison, or exclusion-limit studies.

## Important notes

- The workflow is chunk-based to enable parallel execution on a cluster.
- Large datasets are split into many smaller pieces, each processed separately,
  then merged afterwards.
- This pipeline is intended to convert nuclear recoil inputs into detector-level
  observables, not to be a fully general or production-grade framework.
- It does the job reliably for the project workflow, even though it is not
  especially clean or optimized.

## Typical flow

1. Prepare recoil inputs in chunked form.
2. Run `run_chunk.py` for each chunk (via Slurm).
3. Collect the per-chunk `s1s2` CSV outputs.
4. Merge them into a larger dataset for downstream use.
5. Repeat the same structure for ER background generation.
