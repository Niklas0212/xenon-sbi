# Optuna studies

This folder stores the persistent Optuna study databases used for
hyperparameter optimization. Each SQLite database contains the trials,
parameters, objective values, and trial states for one HPO experiment.

## How studies are created

The study scripts live one directory above this folder:

- `../hpo_script.py` optimizes the `full` or `ntothighest` classifier using
  recoil-spectrum features.
- `../hpo_script_s1s2.py` optimizes the `hist_s1s2` classifier using detector
  level S1/S2 data. Its positional `bg` argument selects the scenario:
  `False` for signal-only and `True` for signal plus background.

Both scripts use a validation loss as the objective and minimize it. They use
Optuna's TPE sampler and median pruner. If a study already exists, the scripts
load it and add the requested number of new trials rather than starting over.

## Database naming

Standard classifier studies use:

```text
<modelname>_<datatag>.db
```

For example, `ntothighest_low.db` stores the optimization study for the
`ntothighest` model and the `low` parameter region. The available model names
are `full` and `ntothighest`; the available data tags are `low`, `mid`, and
`high`.

S1/S2 studies use:

```text
hist_s1s2__signal_only.db
hist_s1s2__signal_bg.db
```

The scenario names correspond to the `bg` argument described above. Existing
databases in this folder may represent completed or partially completed
studies; inspect their trial counts and best parameters before extending them.

## Running HPO

Run commands from the repository root. The jobs are intended to run on the
Midway cluster, typically through Slurm. The standard classifier example is:

```bash
python3 -m hpo.hpo_script --datatag low --modelname ntothighest --n 200
```

For S1/S2 HPO, pass the background scenario as the first positional argument:

```bash
python3 -m hpo.hpo_script_s1s2 False --n 100
python3 -m hpo.hpo_script_s1s2 True --n 100
```

`--n` is the number of additional trials for that execution. The scripts use
`SLURM_CPUS_PER_TASK` for parallel jobs when it is set; otherwise they fall
back to the available CPU count. The PyTorch device is CUDA when available,
and CPU otherwise.

## Required inputs

The standard script expects a generated recoil dataset at:

```text
data/datasets/wimpy/default/wimpy_n300000_<datatag>_default.pt
```

The S1/S2 script expects:

```text
data/datasets/xenon/s1s2/pt/s1s2_n300000_default.pt
data/datasets/xenon/s1s2/ers/s1s2_ers.csv
```

These paths and the default dataset size are defined in the scripts' local
`HPOConfig` classes. Change those values in the scripts when optimizing a
different dataset or halo model.

## Inspecting results

Optuna can inspect a database directly, for example:

```python
import optuna

study = optuna.load_study(
    study_name="ntothighest_low",
    storage="sqlite:///hpo/optuna_studies/ntothighest_low.db",
)
print(study.best_value)
print(study.best_params)
```

The optimization scripts print the best objective value and parameter set at
the end of each run. The databases contain study results only; trained model
checkpoints are not written by these HPO scripts (`ckpt_path=None`).
