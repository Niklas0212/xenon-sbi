# Dark Matter Parameter Inference via Simulation-Based Inference (SBI)

This repository implements neural network-based likelihood-free parameter inference for WIMP dark matter direct detection experiments. The framework enables fast inference of dark matter parameters (mass and coupling) from simulated and experimental recoil spectra using classifier-based Simulation-Based Inference.

## Theoretical Background

### Dark Matter Direct Detection

Dark matter comprises ~85% of the matter in the universe, yet its fundamental nature remains unknown. This project focuses on **Weakly Interacting Massive Particles (WIMPs)**, a leading dark matter candidate. WIMPs can elastically scatter with atomic nuclei, producing nuclear recoils that are detectable in underground experiments.

#### Physics Framework

**Signal Model**: The differential recoil rate for WIMP-nucleus scattering is given by:

$$\frac{d\Gamma}{dE_R} = n_\chi \sigma \int_{v_{min}}^{\infty} \eta(v) \frac{d\sigma}{dE_R} dv$$

where:
- $n_\chi$ = local WIMP number density
- $\sigma$ = cross section
- $E_R$ = nuclear recoil energy
- $v$ = WIMP velocity
- $\eta(v)$ = velocity distribution (Standard Halo Model or variants)

**Key Parameters**:
- $m_\chi$ = WIMP mass (typically 10–1000 GeV)
- $c_p$ = nucleon-level coupling constant
- Halo parameters: velocity dispersion, escape velocity, solar motion direction

**Background**: Includes contributions from cosmic rays, radioactive decays, and coherent neutrino-nucleus scattering ("neutrino floor").

### Simulation-Based Inference (SBI)

Traditional likelihood-based inference is intractable for complex detector physics because computing the likelihood requires expensive numerical integrations. **Simulation-Based Inference** circumvents this by:

1. Sampling dark matter parameters from a prior
2. Running physics simulations for each parameter set
3. Generating synthetic observables
4. Training a neural network to approximate the likelihood ratios

This repository uses **classifier-based SBI**: training neural networks to discriminate between signal and background, enabling posterior inference through likelihood ratio estimation.

### Analytical Posteriors

For validation and comparison, we compute analytical **Poisson posteriors** on parameter grids. These represent the exact posterior under the assumption that observed counts follow a Poisson distribution with fixed expected rates—providing a gold-standard baseline for neural network performance.

## Project Goals

1. **Efficient Parameter Inference**: Develop fast, accurate methods to infer WIMP properties from detector data
2. **Halo Model Robustness**: Test inference robustness across different galactic halo models (SHM, SHM++, LMC, extreme scenarios)
3. **Neural vs. Analytical**: Compare neural network posteriors against analytical Poisson posteriors
4. **Realistic Detectors**: Support real detector models (XENON, WimPyDD) with realistic backgrounds and efficiencies
5. **Hyperparameter Optimization**: Systematically optimize neural network architectures via Optuna

---

## Project Structure

```
dark-matter-sbi/
├── analytical/              # Analytical posterior computation
│   ├── ppg_class.py        # PoissonPosteriorGrid: gridded analytical posteriors
│   ├── lambda_calculation.py # Rate calculations for parameter grids
│   ├── lambda/             # Precomputed expected rate grids (.npz files)
│   └── notebooks/          # Analysis notebooks for analytical methods
│
├── configs/                # Configuration management
│   └── config.py          # MCConfig, parameter ranges, model architectures
│
├── data/                   # Data generation and processing
│   ├── generate_dataset.py # Monte Carlo dataset generation
│   ├── mc_generator.py     # Core physics simulation interface
│   ├── compute_shmpp.py    # SHM++ halo model calculations
│   ├── datasets/           # Generated training/test datasets (.pt files)
│   │   ├── wimpy/         # WimPyDD detector simulations
│   │   └── xenon/         # XENON detector data
│   └── notebooks/          # Data exploration & visualization
│
├── models/                 # Trained neural network checkpoints
│   ├── wimpy/             # WIMP detector models
│   │   ├── default/       # Fixed SHM parameters
│   │   ├── shm/           # Standard Halo Model
│   │   ├── shmpp/         # SHM++ model
│   │   ├── lmc/           # LMC-influenced halo
│   │   ├── x1/            # Extreme high (+5σ)
│   │   └── x2/            # Extreme low (-5σ)
│   └── xenon/             # XENON detector models
│       ├── offline/       # Offline analysis mode
│       └── online/        # Online/trigger-level analysis
│
├── sbi_training/          # Neural network training scripts
│   ├── training_wimpy.py  # WIMP parameter inference training
│   ├── training_s1s2_bkg_online.py   # Detector-specific training (online)
│   ├── training_s1s2_bkg_offline.py  # Detector-specific training (offline)
│   └── __pycache__/
│
├── sbi_notebooks/         # Example inference workflows
│   ├── sbi_wimpy.ipynb    # WIMP inference demonstration
│   └── sbi_s1s2.ipynb     # S1/S2 signal-to-background analysis
│
├── hpo/                    # Hyperparameter optimization
│   ├── hpo_script.py       # Optuna HPO runner
│   ├── optuna_studies/     # Optuna study results
│   └── notebooks/          # HPO analysis & result loading
│
├── performances/           # Performance evaluation & metrics
│   ├── metrics.py          # Classification & Poisson posterior metrics
│   ├── performances.py     # Compute performance statistics
│   ├── results_default/    # Performance results by model
│   └── notebooks/          # Visualization & comparison studies
│
├── utils/                  # Core utilities
│   ├── architectures.py    # MLP models (Full, Hist, Ntot, etc.)
│   ├── training.py         # Training loops & early stopping
│   ├── posteriors.py       # Posterior computation utilities
│   └── processing.py       # Feature preprocessing & data loaders
│
├── WimPyDD/               # External: WimPyDD dark matter simulation package
│   ├── Targets/           # Detector target configurations
│   ├── Halo_functions/    # Galactic halo models
│   └── Experiments/       # Experiment-specific parameters
│
├── felix/                  # Analysis & exploration notebooks (dated)
├── figures/               # Output figures & plots
├── trash/                 # Archived/deprecated code
│
├── testing.ipynb          # Notebook for testing & validation
├── comparison.ipynb       # Benchmark: neural networks vs. analytical posteriors
├── CITATION.cff           # Software citation metadata
├── LICENSE
└── README.md             # This file
```

---

## Software Architecture

### Core Physics Simulation

- **`config.py`**: Parametric configuration
  - `MCConfig`: Monte Carlo simulation parameters (exposure, energy ranges, binning)
  - `PARAM_RANGES`: Prior ranges for $\log_{10}(m_\chi)$ and $\log_{10}(c_p)$ across three energy regions (low, mid, high)
  - `MODEL_CONFIG`: Neural network architecture blueprints (ResNet-style, stacked MLPs, etc.)

- **`mc_generator.py`**: Interface to physics simulator
  - Wraps WimPyDD simulations
  - Supports multiple halo models ("default", "shm", "shmpp", "lmc", "x1", "x2")
  - Returns expected recoil spectra (λ) per parameter tuple

- **`generate_dataset.py`**: Dataset creation pipeline
  - Samples $(\log m_\chi, \log c_p)$ uniformly from priors
  - Calls `mc_generator` for each sample
  - Generates "signal" and "background" event counts
  - Computes Top-K statistics (top 10 observations for dimensionality reduction)
  - Saves as PyTorch tensors (.pt format) for efficient I/O

### Neural Network Training

- **`architectures.py`**: Model implementations
  - `Full_MLP`: Full recoil spectrum as input
  - `Hist_MLP`: Binned histogram representation
  - `Ntot_MLP`: Total event count only
  - `HistS1S2_MLP`: Detector-native S1/S2 signals (light & ionization)
  - `Ntot_Highest_MLP`: Variants with residual connections

- **`training.py`**: Training framework
  - Classifier loss: binary cross-entropy (signal vs. background discrimination)
  - Early stopping with configurable patience
  - Checkpoint management & best model recovery
  - Optional Optuna integration for hyperparameter sweeps
  - Metrics: accuracy, AUC, loss curves

- **`processing.py`**: Data pipeline
  - Preprocessing: normalization, feature engineering
  - PyTorch DataLoaders with train/val/test splits
  - Negative pair generation (ranking-based contrastive learning)

### Inference & Posteriors

- **`posteriors.py`**: Posterior computation utilities
  - Convert classifier scores to likelihood ratios
  - Importance sampler for posterior approximation
  - Credible region extraction

- **`ppg_class.py`** (Analytical): `PoissonPosteriorGrid`
  - Precomputs expected event counts on a 2D parameter grid
  - Evaluates analytical Poisson likelihood for observed counts
  - Computes mode, credible intervals
  - Provides ground truth for neural network validation

- **`lambda_calculation.py`**: Rate grid computation
  - Precomputes $\lambda(\log m_\chi, \log c_p)$ via WimPyDD
  - Caches results in `.npz` format for fast retrieval

### Hyperparameter Optimization

- **`hpo_script.py`**: Optuna integration
  - Automated search over architecture, learning rate, batch size, etc.
  - Study persistence in `optuna_studies/`
  - Cross-validation against test set
  - Early stopping via trial pruners

### Metrics & Performance Evaluation

- **`metrics.py`**: Quantitative metrics
  - Coverage: do credible intervals contain true parameters?
  - Bias & MSE of posterior means
  - Calibration: do posterior credibilities match empirical rates?
  - Comparison between neural and analytical posteriors

- **`performances.py`**: Aggregate performance statistics
  - Batch evaluation across models/halos/regions
  - Result serialization & comparison tables

---

## Key Design Patterns

### 1. **Modular Halo Models**
All halo variations inherit from a common interface in `mc_generator.py`, enabling systematic comparison of robustness.

### 2. **Classification-Based Likelihood Ratio**
Rather than regressing parameters directly, networks learn to discriminate signal from background. The likelihood ratio is recovered from classifier confidence:

$$L(x|\theta) \propto \frac{p_{\text{signal}}(x|\theta)}{p_{\text{background}}(x)}$$

### 3. **Top-K Dimensionality Reduction**
For efficiency, full recoil spectra are compressed to top 10 event energies plus total count, reducing dimensionality while preserving discrimination power.

### 4. **Grid-Based Analytical Baselines**
Precomputed Poisson posterior grids provide exact posteriors (under Poisson assumption) for validation—every neural network result is benchmarked against ground truth.

### 5. **Detector-Agnostic Design**
By wrapping WimPyDD, the framework supports XENON, WIMPY, and other detectors interchangeably through detector configuration files.

---

## Quick Start

### 1. Generate Datasets
```bash
python3 -m data.generate_dataset \
  --n_train 200000 \
  --datatag low \
  --halo_option shm
```

### 2. Train Neural Network Classifier
```bash
python3 -m sbi_training.training_wimpy \
  --n_train 200000 \
  --modelname full \
  --datatag low \
  --halos shm shmpp
```

### 3. Evaluate & Compare
Open `comparison.ipynb` for side-by-side neural network vs. analytical posterior comparison.

---

## Output & Results

- **`models/`**: Trained model checkpoints (.pt format)
- **`performances/results_default/`**: Performance metrics (coverage, bias, etc.)
- **`figures/`**: Publication-ready plots
  - `dev/`: Development/debugging figures
  - `final/`: Final thesis figures
  - `needle/`: Diagnostic plots

---

## References & External Packages

- **WimPyDD**: Core physics simulation (galactic halo, detector response, neutrino backgrounds)
- **PyTorch**: Neural network training & inference
- **Optuna**: Hyperparameter optimization framework
- **Scipy/NumPy**: Numerical utilities & data handling

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{DarkMatterSBI,
  title = {Dark Matter Parameter Inference via Simulation-Based Inference},
  year = {2026},
  url = {https://github.com/[your-repo]},
  note = {Master Thesis Implementation}
}
```

See `CITATION.cff` for additional metadata.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

