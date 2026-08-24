"""
Evaluation metrics for neural network posteriors in dark matter direct detection.

This module provides comprehensive tools to evaluate and visualize neural network
posteriors for WIMP parameter inference (mass and coupling). Implemented metrics:

- Coverage tests using HPD (Highest Posterior Density) regions
- Jensen-Shannon divergence (JSD)
- Wasserstein distances (marginal and sliced 2D)
- Posterior quality metrics (Euclidean, Mahalanobis distances)

Organization:
    1. Coverage Tests (General)
    2. Coverage Tests (Halo/S1S2 Comparisons)
    3. Posterior Quality Metrics
    4. Jensen-Shannon Divergence
    5. Wasserstein Distance Metrics
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import os
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from scipy import stats
from scipy.special import rel_entr
from scipy.ndimage import generic_filter
from scipy.stats import wasserstein_distance

from utils.posteriors import posterior_grid
from utils.processing import (
    preprocess_features,
    load_matching_pairs,
    filter_by_parameter_ranges,
)
from configs.config import PARAM_RANGES


# ============================================================================
# Coverage Tests - Core Functions
# ============================================================================

def compute_credible_region(
    posterior: torch.Tensor,
    levels: np.ndarray = np.linspace(0, 1, 50),
) -> Dict[float, np.ndarray]:
    """
    Compute boolean masks for highest-posterior-density (HPD) regions.
    
    Parameters
    ----------
    posterior : torch.Tensor
        2D posterior probability distribution (will be normalized).
    levels : np.ndarray
        Credibility levels to compute (e.g., np.array([0.68, 0.90]) for 68% and 90%).
    
    Returns
    -------
    Dict[float, np.ndarray]
        Dictionary mapping each level to a boolean mask indicating HPD region.
    """
    # Normalize posterior
    posterior = posterior / posterior.sum()
    
    # Flatten and sort by probability density
    flat = posterior.flatten()
    sorted_vals, _ = torch.sort(flat, descending=True)
    cumsum = torch.cumsum(sorted_vals, 0)
    cumsum = cumsum / cumsum[-1].clone()

    # Compute HPD regions for each level
    regions = {}
    for level in levels:
        idx = torch.searchsorted(cumsum, torch.tensor(level, dtype=torch.float32))
        threshold = sorted_vals[idx]
        regions[level] = (posterior >= threshold).cpu().numpy()
    
    return regions


def coverage_test(
    model: torch.nn.Module,
    features: torch.Tensor,
    thetas: torch.Tensor,
    logm_range: Tuple[float, float],
    logcp_range: Tuple[float, float],
    posteriorbins: int = 100,
    device: str = "cpu",
    levels: np.ndarray = np.linspace(0, 1, 50),
    n_samples: int = 500,
) -> Tuple[Dict[float, float], float, float]:
    """
    Perform coverage test on test data using HPD credible regions.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model.
    features : torch.Tensor
        Features/observations.
    thetas : torch.Tensor
        True parameter values corresponding to features.
    logm_range : Tuple[float, float]
        (min, max) for log10(mass) parameter space.
    logcp_range : Tuple[float, float]
        (min, max) for log10(coupling) parameter space.
    posteriorbins : int
        Number of bins for posterior grid.
    device : str
        Device for computation ('cpu' or 'cuda').
    levels : np.ndarray
        Coverage levels to test (e.g., np.array([0.68, 0.90]) or np.linspace(0, 1, 50)).
    n_samples : int
        Number of test samples to evaluate.
    
    Returns
    -------
    coverages : Dict[float, float]
        Empirical coverage for each nominal level.
    score_abs : float
        Mean absolute deviation from nominal coverage.
    score_signed : float
        Mean signed deviation from nominal coverage.
    """
    n_total = min(n_samples, len(features))
    inside_counts = {level: 0 for level in levels}

    for i in range(n_total):
        counts = features[i].to(device)
        true_theta = thetas[i].cpu().numpy()

        # Preprocess and compute posterior
        counts = preprocess_features(model, counts, device)
        posterior, logm_vals, logcp_vals = posterior_grid(
            counts, logm_range, logcp_range, posteriorbins, model, device
        )

        # Compute HPD regions
        regions = compute_credible_region(posterior, levels)
        
        # Find true parameter location in grid
        logm_true, logcp_true = true_theta
        m_idx = np.argmin(np.abs(logm_vals - logm_true))
        cp_idx = np.argmin(np.abs(logcp_vals - logcp_true))

        # Check if true parameter is inside each HPD region
        for level in levels:
            if regions[level][cp_idx, m_idx]:
                inside_counts[level] += 1

    # Compute empirical coverages
    coverages = {level: inside_counts[level] / n_total for level in levels}

    # Aggregate into scalar metrics
    diffs = [coverages[level] - level for level in levels]
    score_abs = float(np.mean([abs(d) for d in diffs]))
    score_signed = float(np.mean(diffs))

    return coverages, score_abs, score_signed


def plot_coverage(
    coverages: Dict[float, float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot empirical vs. nominal coverage curve.
    
    Parameters
    ----------
    coverages : Dict[float, float]
        Mapping from nominal coverage level to empirical coverage.
    title : Optional[str]
        Deprecated and ignored (kept for backward compatibility).
    save_path : Optional[str]
        Path where the figure is saved as PDF. If extension is missing,
        '.pdf' is appended automatically.
    """
    # Sort levels to guarantee monotonic x-axis ordering.
    levels = np.array(sorted(coverages.keys()))
    empirical = np.array([coverages[level] for level in levels])

    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot ideal first, then empirical on top with markers so both remain visible
    # even when the two curves are nearly identical.
    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=(0, (7, 4)),
        label="Ideal",
        linewidth=4.0,
        alpha=0.9,
        zorder=1,
    )
    ax.plot(
        levels,
        empirical,
        label="Empirical",
        linewidth=5.0,
        color="tab:blue",
        linestyle="-",
        alpha=0.9,
        zorder=3,
    )

    # Thesis-ready typography.
    axis_label_size = 34
    tick_label_size = 28
    legend_size = 27

    ax.set_xlabel("Nominal coverage level", fontsize=axis_label_size)
    ax.set_ylabel("Empirical coverage", fontsize=axis_label_size)
    ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
    ax.grid(True, linestyle="--", linewidth=1.3, alpha=0.7)
    ax.legend(fontsize=legend_size, loc="lower right")

    # Keep title argument for compatibility, but intentionally omit title by default.
    _ = title

    plt.tight_layout()

    if save_path is not None:
        if not save_path.lower().endswith(".pdf"):
            save_path = f"{save_path}.pdf"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", format="pdf")

    plt.show()



# ============================================================================
# Coverage tests - Halo comparisons
# ============================================================================

def plot_coverage_row(
    coverages_list: List[Dict[float, float]],
    labels: List[str],
    title: Optional[str] = None,
    halo_choice: Optional[str] = None,
    score_abs_list: Optional[List[float]] = None,
    score_signed_list: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot coverage curves for multiple models side-by-side.

    Parameters
    ----------
    coverages_list : List[Dict[float, float]]
        List of coverage dictionaries (nominal level -> empirical coverage).
    labels : List[str]
        Model labels for each subplot.
    title : Optional[str]
        Overall figure title.
    save_path : Optional[str]
        Optional path to save the figure. If no extension is provided,
        '.pdf' is appended automatically.
    """
    axis_label_size = 26
    tick_label_size = 22
    panel_title_size = 26
    legend_size = 22
    score_box_size = 20

    n_models = len(coverages_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)

    if n_models == 1:
        axes = [axes]

    for idx, (ax, coverages, label) in enumerate(zip(axes, coverages_list, labels)):
        levels = np.array(list(coverages.keys()))
        empirical = np.array(list(coverages.values()))

        ax.plot(levels, empirical, label="Empirical", linewidth=4.0, color="tab:blue", alpha=0.9)
        ax.plot([0, 1], [0, 1], "k--", label="Ideal", linewidth=3.6)
        ax.set_xlabel("Nominal coverage", fontsize=axis_label_size)
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
        if label == halo_choice:
            ax.set_title(rf"$\bf{{{label}}}$", fontsize=panel_title_size)
        else: 
            ax.set_title(label, fontsize=panel_title_size)
        ax.grid(True, linestyle="--", alpha=0.6)

        if score_abs_list is not None and score_signed_list is not None:
            score_abs = score_abs_list[idx]
            score_signed = score_signed_list[idx]
            ax.text(
                0.96,
                0.04,
                f"CvgAbs={score_abs:.3f}\nCvgSgn={score_signed:.3f}",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=score_box_size,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    axes[0].set_ylabel("Empirical coverage", fontsize=axis_label_size)
    
    if title is not None:
        fig.suptitle(title, fontsize=14)

    axes[0].legend(fontsize=legend_size, loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        if not save_path.lower().endswith(".pdf"):
            save_path = f"{save_path}.pdf"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")

    plt.show()


def coverage_halo_comparison(
    models,
    labels,
    halo_choice,
    datatag,
    n_train,
    logm_range,
    logcp_range,
    device="cpu",
    posteriorbins=100,
    levels=np.linspace(0, 1, 50),
    n_samples=500,
    split="val",
    logm_window=None,
    logcp_window=None,
    savepath: Optional[str] = None,
):
    """
    Side-by-side coverage comparison for multiple halo models.
    
    Parameters
    ----------
    models : list
        List of trained models to compare.
    labels : list[str]
        Labels for each model.
    halo_choice : str
        Halo configuration name.
    datatag : str
        Data category ('low', 'mid', 'high').
    n_train : int
        Number of training samples.
    logm_range : tuple
        (min, max) for log10(mass) posterior grid.
    logcp_range : tuple
        (min, max) for log10(coupling) posterior grid.
    device : str
        Device for computation ('cpu' or 'cuda').
    posteriorbins : int
        Posterior grid resolution.
    levels : np.ndarray
        Coverage levels to test (e.g., np.linspace(0, 1, 50)).
    n_samples : int
        Number of samples to evaluate.
    split : str
        Which data split to use: 'train', 'val', or 'test' (default: 'val').
    logm_window : tuple, optional
        Parameter range filter for mass.
    logcp_window : tuple, optional
        Parameter range filter for coupling.
    savepath : str, optional
        Optional output path for saving the coverage comparison figure.
    """

    # Load data from specified split
    X_test, T_test = load_matching_pairs(
        halo_choice=halo_choice,
        n_train=n_train,
        datatag=datatag,
        split=split,
    )

    # --- Apply parameter window filtering ---
    if logm_window is not None or logcp_window is not None:
        X_test, T_test = filter_by_parameter_ranges(
            T=T_test,
            X=X_test,
            logm_window=logm_window,
            logcp_window=logcp_window,
        )

        print(f"Filtered test samples: {len(T_test)}")

    else:
        print(f"Total test samples: {len(T_test)}")

    coverages_list = []
    score_abs_list = []
    score_signed_list = []

    for model, label in zip(models, labels):
        print(f"Running coverage test for: {label}")

        coverages, score_abs, score_signed = coverage_test(
            model,
            X_test,
            T_test,
            logm_range,
            logcp_range,
            posteriorbins=posteriorbins,
            device=device,
            levels=levels,
            n_samples=n_samples,
        )

        coverages_list.append(coverages)
        score_abs_list.append(score_abs)
        score_signed_list.append(score_signed)

    plot_coverage_row(
        coverages_list,
        labels,
        title=None,
        halo_choice=halo_choice,
        score_abs_list=score_abs_list,
        score_signed_list=score_signed_list,
        save_path=savepath,
    )




# ============================================================================
# Posterior quality metrics 
# ============================================================================

def _posterior_mean_and_cov(
    posterior: torch.Tensor,
    logm: torch.Tensor,
    logcp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and covariance of posterior distribution.
    
    Parameters
    ----------
    posterior : torch.Tensor
        2D posterior probability distribution.
    logm : torch.Tensor
        1D array of log10(mass) grid values.
    logcp : torch.Tensor
        1D array of log10(coupling) grid values.
    
    Returns
    -------
    mean : torch.Tensor
        Posterior mean [mean_logm, mean_logcp].
    cov : torch.Tensor
        2x2 covariance matrix.
    """
    # Ensure torch tensors
    if not torch.is_tensor(logm):
        logm = torch.tensor(logm, dtype=posterior.dtype)
    if not torch.is_tensor(logcp):
        logcp = torch.tensor(logcp, dtype=posterior.dtype)

    # Normalize posterior
    P = posterior / posterior.sum()

    # Create 2D meshgrid
    M, C = torch.meshgrid(logm, logcp, indexing="xy")
    
    # Compute means
    mean_m = (P * M).sum()
    mean_cp = (P * C).sum()

    # Compute covariance
    dm = M - mean_m
    dcp = C - mean_cp

    cov_mm = (P * dm * dm).sum()
    cov_cp = (P * dcp * dcp).sum()
    cov_mcp = (P * dm * dcp).sum()

    cov = torch.tensor([
        [cov_mm, cov_mcp],
        [cov_mcp, cov_cp],
    ])

    mean = torch.tensor([mean_m, mean_cp])
    
    return mean, cov


def _compute_sample_distances(
    model: torch.nn.Module,
    x: torch.Tensor,
    theta_true: torch.Tensor,
    logm_range: Tuple[float, float],
    logcp_range: Tuple[float, float],
    posteriorbins: int,
    device: str,
) -> Tuple[float, float, float]:
    """
    Compute Euclidean, Mahalanobis distance, and posterior volume for a single sample.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model.
    x : torch.Tensor
        Single observation (preprocessed or raw).
    theta_true : torch.Tensor
        True parameters for this observation.
    logm_range, logcp_range : Tuple[float, float]
        Parameter space ranges.
    posteriorbins : int
        Posterior grid resolution.
    device : str
        Device for computation.
    
    Returns
    -------
    d_eucl : float
        Euclidean distance between MAP and true parameters.
    d_maha : float
        Mahalanobis distance (may be NaN on singular covariance).
    volume : float
        Posterior covariance volume (sqrt of determinant).
    """
    # Preprocess features
    x = preprocess_features(model, x, device=device)
    
    # Compute posterior grid
    posterior, logm_grid, logcp_grid = posterior_grid(
        counts=x,
        logm_range=logm_range,
        logcp_range=logcp_range,
        posteriorbins=posteriorbins,
        model=model,
        device=device,
    )
    
    # Compute MAP estimate
    idx = torch.argmax(posterior).item()
    i_cp, i_m = np.unravel_index(idx, posterior.shape)
    theta_map = torch.tensor(
        [logm_grid[i_m], logcp_grid[i_cp]],
        dtype=theta_true.dtype,
        device=device
    )
    
    # Compute posterior mean & covariance
    mean, cov = _posterior_mean_and_cov(posterior, logm_grid, logcp_grid)
    cov = cov.to(device)
    cov += 1e-6 * torch.eye(2, dtype=theta_true.dtype, device=device)
    
    # Compute distances
    diff = theta_map - theta_true
    d_eucl = torch.norm(diff).item()
    
    try:
        d_maha = torch.sqrt(diff @ torch.linalg.inv(cov) @ diff).item()
    except:
        d_maha = np.nan
    
    volume = torch.sqrt(torch.abs(torch.det(cov))).item()
    
    return d_eucl, d_maha, volume


def euclidean_mahalanobis_eval(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    T_test: torch.Tensor,
    logm_range: Tuple[float, float],
    logcp_range: Tuple[float, float],
    posteriorbins: int = 100,
    device: str = "cpu",
    n_samples: Optional[int] = None,
    exclude_zero_event_spectra: bool = True,
) -> Dict[str, float]:
    """
    Evaluate Euclidean and Mahalanobis distances for posterior quality.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model.
    X_test : torch.Tensor
        Test observations of shape (n_test, feature_dim).
    T_test : torch.Tensor
        True parameters of shape (n_test, 2) where [:, 0] = log10(mass), [:, 1] = log10(coupling).
    logm_range : Tuple[float, float]
        (min, max) for log10(mass) parameter space.
    logcp_range : Tuple[float, float]
        (min, max) for log10(coupling) parameter space.
    posteriorbins : int
        Number of bins for posterior grid. Default: 100.
    device : str
        Device for computation (e.g., 'cpu', 'cuda'). Default: 'cpu'.
    n_samples : Optional[int]
        Number of test samples to evaluate. If None, evaluates all non-zero samples. Default: None.

    Returns
    -------
    results : Dict[str, float]
        Dictionary containing:
        - "median_euclidean": Median Euclidean distance across samples
        - "mean_euclidean": Mean Euclidean distance across samples
        - "median_mahalanobis": Median Mahalanobis distance across samples
        - "mean_mahalanobis": Mean Mahalanobis distance across samples
        - "distances_euclidean": All Euclidean distances
        - "distances_mahalanobis": All Mahalanobis distances
        - "logm": List of true log10(mass) values for each sample
        - "logcp": List of true log10(coupling) values for each sample
    """
    euclidean_distances = []
    mahalanobis_distances = []
    logm_vals = []
    logcp_vals = []

    # Filter out zero-event spectra (distances are not meaningful for exclusion regions)
    if exclude_zero_event_spectra:
        zero_mask = X_test.sum(dim=1) == 0
        X_test, T_test = X_test[~zero_mask], T_test[~zero_mask]
    
    n_total = len(X_test) if n_samples is None else min(n_samples, len(X_test))
    
    for i in range(n_total):
        x = X_test[i].to(device)
        theta_true = T_test[i].to(device)
        logm_vals.append(theta_true[0].item())
        logcp_vals.append(theta_true[1].item())
        
        d_eucl, d_maha, _ = _compute_sample_distances(
            model, x, theta_true, logm_range, logcp_range, posteriorbins, device
        )
        
        euclidean_distances.append(d_eucl)
        mahalanobis_distances.append(d_maha)

    
    # Convert to numpy arrays and compute statistics
    euclidean_distances = np.array(euclidean_distances)
    mahalanobis_distances = np.array(mahalanobis_distances)
    
    results = {
        "median_euclidean": float(np.median(euclidean_distances)),
        "mean_euclidean": float(np.mean(euclidean_distances)),
        "median_mahalanobis": float(np.median(mahalanobis_distances)),
        "mean_mahalanobis": float(np.mean(mahalanobis_distances)),
        "distances_euclidean": euclidean_distances,
        "distances_mahalanobis": mahalanobis_distances,
        "logm": logm_vals,
        "logcp": logcp_vals
    }
    
    return results


def plot_mahalanobis_diagnostics(
    results: Dict[str, np.ndarray],
    distance_type: str = "mahalanobis",
    bins: int = 15,
    cmap: str = "viridis",
) -> None:
    """
    Visualize Euclidean or Mahalanobis distances across parameter space.
    
    Parameters
    ----------
    results : Dict[str, np.ndarray]
        Output of euclidean_mahalanobis_eval with keys ["distances_euclidean", 
        "distances_mahalanobis", "logm", "logcp"].
    distance_type : str
        Type of distance to plot ("euclidean" or "mahalanobis").
    bins : int
        Number of bins per axis for heatmap.
    fill_holes : bool
        Whether to fill empty bins in heatmap with neighbor averages.
    cmap : str
        Colormap for plots.
    """
    logm = np.array(results["logm"])
    logcp = np.array(results["logcp"])
    
    if distance_type == "euclidean":
        distances = results["distances_euclidean"]
        title_prefix = "Euclidean"
        label = "Euclidean Distance"
    else:
        distances = results["distances_mahalanobis"]
        title_prefix = "Mahalanobis"
        label = "Mahalanobis Distance"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Raw scatter plot
    sc = axes[0].scatter(logm, logcp, c=distances, cmap=cmap, s=20)
    axes[0].set_title(f"Scatter: Per-sample {title_prefix} Distance")
    axes[0].set_xlabel(r"log$_{10}$(mass)")
    axes[0].set_ylabel(r"log$_{10}$(coupling)")
    plt.colorbar(sc, ax=axes[0], label=label)

    # 2. Histogram
    axes[1].hist(distances, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].set_title(f"Histogram: {title_prefix} Distance Distribution")
    axes[1].set_xlabel(label)
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.mean(distances), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(distances):.3f}")
    axes[1].axvline(np.median(distances), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(distances):.3f}")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_posterior_quality_halo(
    models,
    labels,
    halo_choice,
    datatag,
    n_train,
    device="cpu",
    posteriorbins=100,
    n_samples=500,
    split="val",
    logm_window=None,
    logcp_window=None,
    plotting=False,
    plotting_index=0,
):
    """
    Evaluate posterior quality metrics for multiple halo models.
    
    Computes Euclidean distance, Mahalanobis distance, and posterior covariance
    volume for each model on halo model data. Optionally plots posteriors side-by-side.
    
    Parameters
    ----------
    models : list
        List of trained models.
    labels : list[str]
        Model labels.
    halo_choice : str
        Halo configuration name.
    datatag : str
        Data category ('low', 'mid', 'high').
    n_train : int
        Number of training samples.
    device : str
        Device for computation.
    posteriorbins : int
        Posterior grid resolution.
    n_samples : int
        Number of samples to evaluate.
    split : str
        Which data split to use: 'train', 'val', or 'test' (default: 'val').
    logm_window : tuple, optional
        Parameter range filter for mass.
    logcp_window : tuple, optional
        Parameter range filter for coupling.
    plotting : bool
        Whether to plot posteriors for a single sample.
    plotting_index : int
        Index of sample to plot (if plotting=True).
    
    Returns
    -------
    results : dict
        Performance metrics for each model.
    """

    # Parameter ranges
    logm_range = PARAM_RANGES[datatag]["logm_range"]
    logcp_range = PARAM_RANGES[datatag]["logcp_range"]

    # Load data from specified split
    X_test, T_test = load_matching_pairs(
        halo_choice=halo_choice,
        n_train=n_train,
        datatag=datatag,
        split=split,
    )

    # Apply parameter window filtering
    if logm_window or logcp_window:
        X_test, T_test = filter_by_parameter_ranges(
            T=T_test,
            X=X_test,
            logm_window=logm_window,
            logcp_window=logcp_window,
        )
        print(f"Filtered test samples: {len(T_test)}")

    # Filter out zero-event spectra
    zero_mask = X_test.sum(dim=1) == 0
    X_test, T_test = X_test[~zero_mask], T_test[~zero_mask]
    print(f"Non-zero test samples: {len(X_test)}")

    # Loop over models
    results = {}
    for model_idx, (model, label) in enumerate(zip(models, labels)):
            
        print(f"Evaluating model: {label}")
        metrics = defaultdict(list)

        for i in range(min(n_samples, len(X_test))):

            x, theta_true = X_test[i].to(device), T_test[i].to(device)

            # Compute distances using helper function
            d_eucl, d_maha, volume = _compute_sample_distances(
                model, x, theta_true, logm_range, logcp_range, posteriorbins, device
            )
            
            # --- Plotting ---
            if plotting and i == plotting_index:
                if model_idx == 0:
                    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5), squeeze=False)

                # Get posterior for plotting
                x_preproc = preprocess_features(model, x, device=device)
                posterior, logm_grid, logcp_grid = posterior_grid(
                    counts=x_preproc,
                    logm_range=logm_range,
                    logcp_range=logcp_range,
                    posteriorbins=posteriorbins,
                    model=model,
                    device=device,
                )

                ax = axes[0, model_idx]
                im = ax.contourf(10**logm_grid, 10**logcp_grid, posterior.cpu(), levels=50, cmap="viridis")
                ax.scatter(10**theta_true[0], 10**theta_true[1], color="red", s=50)
                ax.set_xscale("log"); ax.set_yscale("log")
                if label == halo_choice:
                    ax.set_title(rf"$\bf{{{labels[model_idx]}}}$")
                else:
                    ax.set_title(labels[model_idx])
                ax.text(
                    0.95,
                    0.20,
                    f"d_eucl={d_eucl:.3f}\nd_maha={d_maha:.3f}\nvol={volume:.3e}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )
                if model_idx == len(models) - 1:
                    fig.suptitle(
                        rf"Posterior comparison for one sample"
                    )
                    plt.tight_layout()
                    plt.show()

            metrics["d_euclidean"].append(d_eucl)
            metrics["d_mahalanobis"].append(d_maha)
            metrics["volume"].append(volume)
            
        results[label] = metrics

    if plotting:
        return results

    return results


def compare_metric_histograms(
    performances: Dict[str, Dict[str, np.ndarray]],
    metric: Optional[str] = None,
    bins: int = 50,
    density: bool = True,
    figsize_per_panel: Tuple[float, float] = (6.8, 4.8),
    reference_label: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    histogram_alpha: float = 0.64,
    decimals: int = 2,
    axis_label_fontsize: int = 24,
    tick_fontsize: int = 18,
    legend_fontsize: int = 16,
    textbox_fontsize: int = 16,
    annotation_fontsize: int = 24,
    line_width: float = 0.0,
    vline_width: float = 6.0,
    savepath: Optional[str] = None,
) -> None:
    """
    Plot histogram comparisons for one or multiple posterior quality metrics.

    Parameters
    ----------
    performances : Dict[str, Dict[str, np.ndarray]]
        Nested dictionary with per-model metric arrays.
    metric : str or None
        If provided (e.g. ``"d_euclidean"``), plot only that metric.
        If ``None`` or ``"all"``, plots all default metrics in one large figure.
    bins : int
        Number of histogram bins.
    density : bool
        Whether to plot density-normalized histograms.
    figsize_per_panel : tuple
        Per-panel figure size scaling (width, height).
    reference_label : str, optional
        Required for volume ratio normalization.
    metrics : list[str], optional
        Explicit metric list to plot. If provided, overrides ``metric``.
    histogram_alpha : float
        Histogram bar alpha value.
    decimals : int
        Number of decimals for med/avg values in the stats text boxes.
    axis_label_fontsize, tick_fontsize, legend_fontsize, textbox_fontsize,
    annotation_fontsize : int
        Font sizes for axis labels, ticks, legends, text boxes, and row annotations.
    line_width : float
        Histogram edge line width.
    vline_width : float
        Vertical summary line width.
    savepath : str, optional
        Optional output file path for saving the figure.
        If no extension is provided, '.pdf' is appended.
    """
    labels = list(performances.keys())
    n_models = len(labels)

    if metrics is not None:
        metrics_to_plot = metrics
    elif metric is None or metric == "all":
        metrics_to_plot = ["d_euclidean", "d_mahalanobis", "volume"]
    else:
        metrics_to_plot = [metric]

    metric_display = {
        "d_euclidean": "Euclidean",
        "d_mahalanobis": "Mahalanobis",
        "volume": "Volume",
    }
    metric_xlabel = {
        "d_euclidean": r"$d_E$",
        "d_mahalanobis": r"$d_M$",
        "volume": r"$V/V_{\mathrm{ref}}$",
    }

    # Blue-forward palette (colorblind-friendly leaning) for less bland panels.
    hist_face_color = "#0046A0"
    hist_edge_color = "#0046A0"
    mean_color = "#B22222"
    median_color = "black"

    n_rows = len(metrics_to_plot)
    fig, axes = plt.subplots(
        n_rows,
        n_models,
        figsize=(figsize_per_panel[0] * n_models, figsize_per_panel[1] * n_rows),
        sharey="row",
    )

    if n_rows == 1 and n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_models == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, metric_name in enumerate(metrics_to_plot):
        if metric_name not in ["d_euclidean", "d_mahalanobis", "volume"]:
            raise ValueError(
                f"Unsupported metric '{metric_name}'. Use one of "
                "['d_euclidean', 'd_mahalanobis', 'volume']."
            )

        reference_values = None
        x_label = metric_xlabel.get(metric_name, metric_name)
        if metric_name == "volume":
            if reference_label is None:
                raise ValueError("reference_label is required when plotting 'volume'")
            if reference_label not in performances:
                raise ValueError(f"reference_label '{reference_label}' not found in performances")
            reference_values = np.asarray(performances[reference_label][metric_name])

        row_values = []
        for label in labels:
            values = np.asarray(performances[label][metric_name]).ravel()
            if reference_values is not None:
                values = values / reference_values
            if metric_name == "volume" and reference_label is not None and label == reference_label:
                continue
            row_values.append(values)

        concatenated = np.concatenate(row_values)
        xmin, xmax = np.min(concatenated), np.max(concatenated)
        if xmin == xmax:
            eps = 1e-8 if xmin == 0 else abs(xmin) * 1e-8
            xmin -= eps
            xmax += eps

        for col_idx, label in enumerate(labels):
            ax = axes[row_idx, col_idx]

            values = np.asarray(performances[label][metric_name]).ravel()
            if reference_values is not None:
                values = values / reference_values

            if metric_name == "volume" and reference_label is not None and label == reference_label:
                ax.set_xlim(xmin, xmax)
                ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
                if col_idx == 0:
                    ax.set_ylabel("Density", fontsize=axis_label_fontsize)
                ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
                ax.tick_params(axis="y", which="both", labelleft=False)
                ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
                ax.set_xticks([])
                ax.set_facecolor("#eef3fb")
                ax.text(
                    0.50,
                    0.50,
                    "reference\n(excluded)",
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    fontsize=textbox_fontsize,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9),
                )

                if row_idx == 0:
                    if reference_label is not None and label == reference_label:
                        ax.set_title(rf"$\bf{{{label}}}$", fontsize=axis_label_fontsize)
                    else:
                        ax.set_title(label, fontsize=axis_label_fontsize)
                else:
                    ax.set_title("")

                if col_idx == n_models - 1:
                    row_name = metric_display.get(metric_name, metric_name)
                    ax.text(
                        1.02,
                        0.50,
                        row_name,
                        transform=ax.transAxes,
                        rotation=-90,
                        va="center",
                        ha="left",
                        fontsize=annotation_fontsize,
                    )
                continue

            mean = values.mean()
            std = values.std()
            sem = std / np.sqrt(len(values)) if len(values) > 0 else np.nan

            median = np.median(values)
            mad = np.median(np.abs(values - median))
            mad_sem = mad / np.sqrt(len(values)) if len(values) > 0 else np.nan

            ax.hist(
                values,
                bins=bins,
                range=(xmin, xmax),
                density=density,
                color=hist_face_color,
                edgecolor=hist_edge_color,
                linewidth=max(line_width, 1.4),
                alpha=histogram_alpha,
            )

            ax.axvline(
                mean,
                color=mean_color,
                linestyle="--",
                linewidth=vline_width,
            )
            ax.axvline(
                median,
                color=median_color,
                linestyle="--",
                linewidth=vline_width,
            )

            ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
            if col_idx == 0:
                ax.set_ylabel("Density", fontsize=axis_label_fontsize)

            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
            ax.tick_params(axis="y", which="both", labelleft=False)
            
            # Set x-axis formatter with appropriate decimals
            if metric_name == "d_euclidean":
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            elif metric_name == "d_mahalanobis":
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            else:  # volume
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            
            # Limit number of ticks to avoid overlapping labels
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

            ax.text(
                0.97,
                0.94,
                f"avg={mean:.{decimals}f}±{sem:.{decimals}f}",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=textbox_fontsize + 2,
                color=mean_color,
                bbox=dict(
                    boxstyle="round,pad=0.45",
                    facecolor=(0.88, 0.94, 1.0),
                    edgecolor=mean_color,
                    linewidth=2.0,
                    alpha=0.95,
                ),
            )
            ax.text(
                0.97,
                0.72,
                f"med={median:.{decimals}f}±{mad_sem:.{decimals}f}",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=textbox_fontsize + 2,
                color=median_color,
                bbox=dict(
                    boxstyle="round,pad=0.45",
                    facecolor=(0.90, 0.95, 1.0),
                    edgecolor=median_color,
                    linewidth=2.0,
                    alpha=0.95,
                ),
            )

            if row_idx == 0:
                if reference_label is not None and label == reference_label:
                    ax.set_title(rf"$\bf{{{label}}}$", fontsize=axis_label_fontsize)
                else:
                    ax.set_title(label, fontsize=axis_label_fontsize)
            else:
                ax.set_title("")

            if col_idx == n_models - 1:
                row_name = metric_display.get(metric_name, metric_name)
                ax.text(
                    1.02,
                    0.50,
                    row_name,
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    ha="left",
                    fontsize=annotation_fontsize,
                )

    plt.tight_layout()

    if savepath is not None:
        if not savepath.lower().endswith(".pdf"):
            savepath = f"{savepath}.pdf"
        save_dir = os.path.dirname(savepath)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight", format="pdf")

    plt.show()





# ============================================================================
# S1S2-specific evaluation functions
# ============================================================================

def coverage_test_s1s2(
    model,
    test_features,
    test_thetas,
    logm_range,
    logcp_range,
    posteriorbins=100,
    device="cpu",
    levels=np.array([0.68, 0.90]),
    n_samples=500,
):
    """
    Perform a coverage test for S1S2 histogram models.
    
    Parameters
    ----------
    model : nn.Module
        Trained S1S2 model.
    test_features : torch.Tensor
        Flattened S1S2 histogram tensors of shape (n_samples, bins*bins).
    test_thetas : torch.Tensor
        True parameters of shape (n_samples, 2) where [:, 0] = log10(mchi), [:, 1] = log10(cp).
    logm_range : tuple
        (logm_min, logm_max) for posterior grid.
    logcp_range : tuple
        (logcp_min, logcp_max) for posterior grid.
    posteriorbins : int
        Number of bins for posterior grid.
    device : str or torch.device
        Device for computation.
    levels : np.ndarray
        Coverage levels to test (e.g., np.array([0.68, 0.90]) or np.linspace(0, 1, 50)).
    n_samples : int
        Number of test samples to use.
    
    Returns
    -------
    coverages : dict
        Map from level -> empirical coverage
    score_abs : float
        Mean absolute deviation from nominal coverage
    score_signed : float
        Mean signed deviation from nominal coverage
    """
    n_total = min(n_samples, len(test_features))
    inside_counts = {lev: 0 for lev in levels}

    for i in range(n_total):
        counts = test_features[i].to(device)  # Shape: (bins*bins,)
        true_theta = test_thetas[i].cpu().numpy()

        # For S1S2, features are already preprocessed (no additional preprocessing needed)
        counts = counts.to(device)
        
        posterior, lmv, lcv = posterior_grid(
            counts, logm_range, logcp_range, posteriorbins, model, device
        )

        regions = compute_credible_region(posterior, levels)
        logm_true, logcp_true = true_theta
        mi = np.argmin(np.abs(lmv - logm_true))
        ci = np.argmin(np.abs(lcv - logcp_true))

        for lev in levels:
            if regions[lev][ci, mi]:  # inside HPD region
                inside_counts[lev] += 1

    coverages = {lev: inside_counts[lev] / n_total for lev in levels}

    # --- Aggregate into scalar metrics ---
    diffs = [coverages[lev] - lev for lev in levels]
    score_abs = float(np.mean([abs(d) for d in diffs]))
    score_signed = float(np.mean(diffs))

    return coverages, score_abs, score_signed


def coverage_comparison_s1s2(
    models,
    labels,
    halo_choice,
    X_test,
    T_test,
    logm_range,
    logcp_range,
    device="cpu",
    posteriorbins=100,
    levels=np.linspace(0, 1, 50),
    n_samples=500,
    logm_window=None,
    logcp_window=None,
    savepath: Optional[str] = None,
):
    """
    Run coverage tests for multiple S1S2 models side-by-side and plot results.
    
    Matches the signature and behavior of coverage_halo_comparison for consistency.
    
    Parameters
    ----------
    models : list[nn.Module]
        List of trained models.
    labels : list[str]
        Model labels for plotting.
    X_test : torch.Tensor
        Test features of shape (n_test, bins*bins).
    T_test : torch.Tensor
        Test parameters of shape (n_test, 2).
    logm_range, logcp_range : tuple
        Parameter ranges for posterior grid.
    device : str or torch.device
        Device for computation.
    posteriorbins : int
        Posterior grid resolution.
    levels : np.ndarray
        Coverage levels to test (e.g., np.linspace(0, 1, 50)).
    n_samples : int
        Number of test samples.
    logm_window : tuple, optional
        Parameter range filter for mass.
    logcp_window : tuple, optional
        Parameter range filter for coupling.
    
    Returns
    -------
    None (plots results directly)
    """
    
    # --- Apply parameter window filtering ---
    if logm_window is not None or logcp_window is not None:
        X_test, T_test = filter_by_parameter_ranges(
            T=T_test,
            X=X_test,
            logm_window=logm_window,
            logcp_window=logcp_window,
        )
        print(f"Filtered test samples: {len(T_test)}")
    else:
        print(f"Total test samples: {len(T_test)}")

    coverages_list = []
    score_abs_list = []
    score_signed_list = []

    for model, label in zip(models, labels):
        print(f"Running coverage test for: {label}")

        coverages, score_abs, score_signed = coverage_test_s1s2(
            model,
            X_test,
            T_test,
            logm_range,
            logcp_range,
            posteriorbins=posteriorbins,
            device=device,
            levels=levels,
            n_samples=n_samples,
        )

        coverages_list.append(coverages)
        score_abs_list.append(score_abs)
        score_signed_list.append(score_signed)

    plot_coverage_row(
        coverages_list,
        labels,
        halo_choice=halo_choice,
        score_abs_list=score_abs_list,
        score_signed_list=score_signed_list,
        save_path=savepath,
    )


def model_performances_s1s2(
    models,
    labels,
    halo_choice,
    X_test,
    T_test,
    logm_range,
    logcp_range,
    device="cpu",
    posteriorbins=100,
    n_samples=500,
    plotting=False,
    plotting_index=0,
):
    """
    Compute performance metrics (Euclidean, Mahalanobis distance, volume) for S1S2 models.
    
    Analogous to evaluate_posterior_quality_halo() but for S1S2 histograms.
    
    Parameters
    ----------
    models : list[nn.Module]
        Trained S1S2 models.
    labels : list[str]
        Model labels.
    X_test : torch.Tensor
        Test features of shape (n_test, bins*bins) -> already flattened.
    T_test : torch.Tensor
        True parameters of shape (n_test, 2).
    logm_range, logcp_range : tuple
        Parameter ranges for posterior grid.
    device : str or torch.device
        Device for computation.
    posteriorbins : int
        Posterior grid resolution.
    n_samples : int
        Number of test samples to evaluate.
    plotting : bool
        Whether to plot posteriors for a single sample.
    plotting_index : int
        Index of sample to plot (if plotting=True).
    
    Returns
    -------
    results : dict
        Performance metrics for each model. Each entry has keys:
        - "d_euclidean": numpy array of Euclidean distances
        - "d_mahalanobis": numpy array of Mahalanobis distances
        - "volume": numpy array of posterior covariance determinants
    """
    # Filter out zero-event spectra (for signal only)
    zero_mask = X_test.sum(dim=1) == 0
    X_test_filtered = X_test[~zero_mask]
    T_test_filtered = T_test[~zero_mask]
    print(f"Non-zero test samples: {len(X_test_filtered)} / {len(X_test)}")

    results = {}
    for model_idx, (model, label) in enumerate(zip(models, labels)):
            
        print(f"Evaluating model: {label}")
        metrics = defaultdict(list)

        n_eval = min(n_samples, len(X_test_filtered))
        for i in range(n_eval):

            x, theta_true = X_test_filtered[i].to(device), T_test_filtered[i].to(device)

            # Compute distances and metrics using helper function
            d_eucl, d_maha, volume = _compute_sample_distances(
                model, x, theta_true, logm_range, logcp_range, posteriorbins, device
            )

            # --- Plotting ---
            if plotting and i == plotting_index:
                if model_idx == 0:
                    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5), squeeze=False)

                # Get posterior for plotting visualization (preprocess first)
                x_preproc = preprocess_features(model, x, device=device)
                posterior, logm_grid, logcp_grid = posterior_grid(
                    counts=x_preproc,
                    logm_range=logm_range,
                    logcp_range=logcp_range,
                    posteriorbins=posteriorbins,
                    model=model,
                    device=device,
                )

                ax = axes[0, model_idx]
                im = ax.contourf(10**logm_grid, 10**logcp_grid, posterior.cpu(), levels=50, cmap="viridis")
                ax.scatter(10**theta_true[0], 10**theta_true[1], color="red", s=50, marker="x")
                ax.set_xscale("log")
                ax.set_yscale("log")
                if label == halo_choice:
                    ax.set_title(rf"$\bf{{{labels[model_idx]}}}$")
                else:
                    ax.set_title(labels[model_idx])
                fig.colorbar(im, ax=ax)
                
                # Display metrics in a box
                ax.text(
                    0.95,
                    0.20,
                    f"d_eucl={d_eucl:.3f}\nd_maha={d_maha:.3f}\nvol={volume:.3e}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )
                
                if model_idx == len(models) - 1:
                    fig.suptitle(f"Posterior comparison for one sample")
                    plt.tight_layout()
                    plt.show()

            metrics["d_euclidean"].append(d_eucl)
            metrics["d_mahalanobis"].append(d_maha)
            metrics["volume"].append(volume)

        # Convert lists to numpy arrays for consistency with wimpy version
        results[label] = {k: np.array(v) for k, v in metrics.items()}

    return results





# ============================================================================
# Jensen-Shannon divergence (JSD)
# ============================================================================

def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute Kullback-Leibler divergence KL(P || Q) for discrete distributions.
    
    Parameters
    ----------
    P : np.ndarray
        First probability distribution (will be normalized).
    Q : np.ndarray
        Second probability distribution (will be normalized).
    
    Returns
    -------
    float
        KL divergence value.
    """
    # Clip and normalize
    P = np.clip(P, 1e-12, None)
    P = P / P.sum()
    Q = np.clip(Q, 1e-12, None)
    Q = Q / Q.sum()
    
    return np.sum(rel_entr(P, Q))


def js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Parameters
    ----------
    P : np.ndarray
        First probability distribution (will be normalized).
    Q : np.ndarray
        Second probability distribution (will be normalized).
    
    Returns
    -------
    float
        JS divergence value (symmetric, bounded in [0, 1] when normalized by log(2)).
    """
    # Clip and normalize
    P = np.clip(P, 1e-12, None)
    P = P / P.sum()
    Q = np.clip(Q, 1e-12, None)
    Q = Q / Q.sum()
    
    # Compute mixture
    M = 0.5 * (P + Q)
    
    # JS divergence as average of KL divergences
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)


def jsd_eval(
    model: torch.nn.Module,
    test_features: torch.Tensor,
    test_thetas: torch.Tensor,
    logm_range: Tuple[float, float],
    logcp_range: Tuple[float, float],
    ppg,  # PoissonPosteriorGrid
    posteriorbins: int = 100,
    device: str = "cpu",
    n_samples: int = 200,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute Jensen-Shannon divergence between NN and analytical posteriors.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model.
    test_features : torch.Tensor
        Test observations.
    test_thetas : torch.Tensor
        True parameters corresponding to test_features.
    logm_range : Tuple[float, float]
        (min, max) for log10(mass) parameter space.
    logcp_range : Tuple[float, float]
        (min, max) for log10(coupling) parameter space.
    ppg : PoissonPosteriorGrid
        Analytical posterior grid for comparison.
    posteriorbins : int
        Number of bins for posterior grid.
    device : str
        Device for computation.
    n_samples : int
        Number of test samples to evaluate.

    Returns
    -------
    mean_jsd : float
        Mean JS divergence across samples.
    results : Dict[str, np.ndarray]
        Dictionary with keys:
        - "jsd": Array of JSD values
        - "logm": Array of log10(mass) values (true)
        - "logcp": Array of log10(coupling) values (true)
    """
    n_total = min(n_samples, len(test_features))
    jsd_values, logm_vals, logcp_vals = [], [], []

    for i in range(n_total):
        counts = test_features[i].to(device)
        logm_true, logcp_true = test_thetas[i].cpu().numpy()

        # Analytical posterior
        counts_np = counts.cpu().numpy()
        counts_analytical = counts_np[:100]
        posterior_true = ppg.posterior_binned(counts_analytical)

        # Learned posterior
        counts = preprocess_features(model, counts, device)
        posterior_learned, _, _ = posterior_grid(
            counts, logm_range, logcp_range, posteriorbins, model, device
        )
        posterior_learned = posterior_learned.cpu().numpy()

        # Compute JS divergence (normalized to [0,1] by dividing by log(2))
        jsd_val = js_divergence(
            posterior_true.flatten(),
            posterior_learned.flatten()
        ) / np.log(2)
        
        jsd_values.append(jsd_val)
        logm_vals.append(logm_true)
        logcp_vals.append(logcp_true)

    mean_jsd = float(np.mean(jsd_values))
    results = {
        "jsd": np.array(jsd_values),
        "logm": np.array(logm_vals),
        "logcp": np.array(logcp_vals),
    }
    
    return mean_jsd, results


def compute_filled_heatmap(
    logm: np.ndarray,
    logcp: np.ndarray,
    jsd: np.ndarray,
    bins: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D heatmap of average JSD, filling empty bins with neighbor averages.
    
    Parameters
    ----------
    logm : np.ndarray
        Array of log10(mass) values.
    logcp : np.ndarray
        Array of log10(coupling) values.
    jsd : np.ndarray
        Array of JSD values.
    bins : int
        Number of bins per axis.
    
    Returns
    -------
    filled_heatmap : np.ndarray
        2D heatmap with NaN values filled.
    xedges : np.ndarray
        Bin edges for mass axis.
    yedges : np.ndarray
        Bin edges for coupling axis.
    """
    # Compute binned statistics
    heatmap, xedges, yedges, binnum = stats.binned_statistic_2d(
        logm, logcp, jsd, statistic="mean", bins=bins
    )

    def nanmean_filter(values):
        """Replace NaN with neighbor mean in a filter window."""
        center = values[len(values) // 2]
        if np.isnan(center):
            neighbors = values[~np.isnan(values)]
            return np.mean(neighbors) if len(neighbors) > 0 else np.nan
        return center

    # Apply 3x3 neighborhood filter
    filled_heatmap = generic_filter(
        heatmap, nanmean_filter, size=3, mode="constant", cval=np.nan
    )

    return filled_heatmap, xedges, yedges


def plot_jsd_diagnostics(
    results: Dict[str, np.ndarray],
    bins: int = 15,
    fill_holes: bool = False,
    cmap: str = "viridis",
) -> None:
    """
    Visualize JSD values across parameter space.
    
    Parameters
    ----------
    results : Dict[str, np.ndarray]
        Output of jsd_eval with keys ["jsd", "logm", "logcp"].
    bins : int
        Number of bins per axis for heatmap.
    fill_holes : bool
        Whether to fill empty bins in heatmap with neighbor averages.
    cmap : str
        Colormap for plots.
    """
    logm = results["logm"]
    logcp = results["logcp"]
    jsd = results["jsd"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Raw scatter plot
    sc = axes[0].scatter(logm, logcp, c=jsd, cmap=cmap, s=20)
    axes[0].set_title("Scatter: Per-sample JSD")
    axes[0].set_xlabel(r"log$_{10}$(mass)")
    axes[0].set_ylabel(r"log$_{10}$(coupling)")
    plt.colorbar(sc, ax=axes[0], label="JSD")

    # 2. Heatmap (binned average)
    if fill_holes:
        heatmap, xedges, yedges = compute_filled_heatmap(logm, logcp, jsd, bins=bins)
    else:
        heatmap, xedges, yedges, _ = stats.binned_statistic_2d(
            logm, logcp, jsd, statistic="mean", bins=bins
        )

    im = axes[1].imshow(
        heatmap.T, origin="lower", aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap
    )
    axes[1].set_title("Heatmap: Avg. JSD per bin")
    axes[1].set_xlabel(r"log$_{10}$(mass)")
    axes[1].set_ylabel(r"log$_{10}$(coupling)")
    plt.colorbar(im, ax=axes[1], label="Mean JSD")

    plt.tight_layout()
    plt.show()



# ============================================================================
# Wasserstein distance - marginal
# ============================================================================

def wasserstein_marginals(P: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    """
    Compute Wasserstein distances for marginals of 2D distributions.

    Parameters
    ----------
    P, Q : np.ndarray
        2D arrays representing discrete probability distributions (same shape).
        Will be normalized internally.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - "mass": Wasserstein distance for marginal over mass axis
        - "cp": Wasserstein distance for marginal over coupling axis
    """
    # Clip and normalize
    P = np.clip(P, 1e-12, None)
    P = P / P.sum()
    Q = np.clip(Q, 1e-12, None)
    Q = Q / Q.sum()

    # Marginalize: sum over each axis
    P_mass, Q_mass = P.sum(axis=0), Q.sum(axis=0)  # Sum over coupling
    P_cp, Q_cp = P.sum(axis=1), Q.sum(axis=1)      # Sum over mass

    # Validate shapes
    assert len(P_mass) == len(Q_mass), \
        f"Mass marginals mismatch: {P_mass.shape} vs {Q_mass.shape}"
    assert len(P_cp) == len(Q_cp), \
        f"Coupling marginals mismatch: {P_cp.shape} vs {Q_cp.shape}"
    
    # Use bin indices as locations
    x_mass = np.arange(len(P_mass))
    x_cp = np.arange(len(P_cp))

    # Compute Wasserstein distances
    d_mass = wasserstein_distance(x_mass, x_mass, P_mass, Q_mass)
    d_cp = wasserstein_distance(x_cp, x_cp, P_cp, Q_cp)

    return {"mass": d_mass, "cp": d_cp}


def wasserstein_eval_marginals(
    model: torch.nn.Module,
    test_features: torch.Tensor,
    test_thetas: torch.Tensor,
    logm_range: Tuple[float, float],
    logcp_range: Tuple[float, float],
    ppg,  # PoissonPosteriorGrid
    posteriorbins: int = 100,
    device: str = "cpu",
    n_samples: int = 200,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Evaluate marginal Wasserstein distances between NN and analytical posteriors.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model.
    test_features : torch.Tensor
        Test observations.
    test_thetas : torch.Tensor
        True parameters.
    logm_range : Tuple[float, float]
        (min, max) for log10(mass).
    logcp_range : Tuple[float, float]
        (min, max) for log10(coupling).
    ppg : PoissonPosteriorGrid
        Analytical posterior grid.
    posteriorbins : int
        Posterior grid resolution.
    device : str
        Device for computation.
    n_samples : int
        Number of samples to evaluate.

    Returns
    -------
    mean_dists : Dict[str, float]
        Mean Wasserstein distances for "mass" and "cp" marginals.
    results : Dict[str, np.ndarray]
        Per-sample distances and true parameters.
    """
    n_total = min(n_samples, len(test_features))
    d_mass_list, d_cp_list = [], []
    logm_vals, logcp_vals = [], []

    for i in range(n_total):
        counts = test_features[i].to(device)
        logm_true, logcp_true = test_thetas[i].cpu().numpy()

        # Analytical posterior
        counts_np = counts.cpu().numpy()
        counts_analytical = counts_np[:100]
        posterior_true = ppg.posterior_binned(counts_analytical)

        # Learned posterior
        counts = preprocess_features(model, counts, device)
        posterior_learned, _, _ = posterior_grid(
            counts, logm_range, logcp_range, posteriorbins, model, device
        )
        posterior_learned = posterior_learned.cpu().numpy()

        # Compute marginal Wasserstein distances
        dists = wasserstein_marginals(posterior_true, posterior_learned)
        d_mass_list.append(dists["mass"])
        d_cp_list.append(dists["cp"])
        logm_vals.append(logm_true)
        logcp_vals.append(logcp_true)

    mean_dists = {
        "mass": float(np.mean(d_mass_list)),
        "cp": float(np.mean(d_cp_list)),
    }
    
    results = {
        "mass": np.array(d_mass_list),
        "cp": np.array(d_cp_list),
        "logm": np.array(logm_vals),
        "logcp": np.array(logcp_vals),
    }

    return mean_dists, results


def plot_wasserstein_marginals(results, mean_dists, bins=15, cmap="viridis", fill_holes=False):
    """
    Plot marginal Wasserstein distances for mass and coupling.

    Parameters
    ----------
    results : dict
        Output of `wasserstein_eval_marginals` with keys
        ["mass", "cp", "logm", "logcp"].
    mean_dists : dict
        Mean Wasserstein distances {"mass": float, "cp": float}.
    savepath : str or None
        If given, save figure to this path.
    """

    logm  = results["logm"]
    logcp = results["logcp"]
    d_mass = results["mass"]
    d_cp   = results["cp"]
    d_avg  = (d_mass + d_cp) / 2
    
    # -------------------------
    # Figure 1: 1D Scatter plots (distance vs. parameter)
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(logm, d_mass, alpha=0.7, s=20)
    axes[0].set_xlabel("True log10(mass)")
    axes[0].set_ylabel("Wasserstein distance (mass)")
    axes[0].set_title(f"Mass marginal (mean = {mean_dists['mass']:.4f})")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].scatter(logcp, d_cp, alpha=0.7, color="orange", s=20)
    axes[1].set_xlabel("True log10(coupling)")
    axes[1].set_ylabel("Wasserstein distance (cp)")
    axes[1].set_title(f"Coupling marginal (mean = {mean_dists['cp']:.4f})")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()

    # -------------------------
    # Figure 2: 2D Scatter plots (parameter space colored by distance)
    # -------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    sc0 = axes2[0].scatter(logm, logcp, c=d_mass, cmap=cmap, s=20)
    axes2[0].set_xlabel("True log10(mass)")
    axes2[0].set_ylabel("True log10(coupling)")
    axes2[0].set_title("Mass marginal distance")
    plt.colorbar(sc0, ax=axes2[0], label="Wasserstein (mass)")

    sc1 = axes2[1].scatter(logm, logcp, c=d_cp, cmap=cmap, s=20)
    axes2[1].set_xlabel("True log10(mass)")
    axes2[1].set_ylabel("True log10(coupling)")
    axes2[1].set_title("Coupling marginal distance")
    plt.colorbar(sc1, ax=axes2[1], label="Wasserstein (cp)")
    fig2.tight_layout()
    plt.show()



# ============================================================================
# Wasserstein distance - sliced
# ============================================================================

def sliced_wasserstein_2D(
    P: np.ndarray,
    Q: np.ndarray,
    n_projections: int = 50,
    seed: Optional[int] = None,
) -> float:
    """
    Approximate 2D Wasserstein distance via sliced Wasserstein method.
    
    Parameters
    ----------
    P, Q : np.ndarray
        2D discrete probability distributions (must have same shape).
    n_projections : int
        Number of random projection directions.
    seed : Optional[int]
        Random seed for reproducibility.
    
    Returns
    -------
    float
        Approximate sliced Wasserstein distance.
    """
    rng = np.random.default_rng(seed)
    
    # Normalize to probabilities
    P = np.clip(P, 1e-12, None)
    P = P / P.sum()
    Q = np.clip(Q, 1e-12, None)
    Q = Q / Q.sum()

    # Get coordinates of pixels
    H, W = P.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # Shape: (H*W, 2)

    # Flatten probability distributions
    P_flat = P.ravel()
    Q_flat = Q.ravel()

    # Compute Wasserstein distance along random projections
    distances = []
    for _ in range(n_projections):
        # Random unit direction in 2D
        theta = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])

        # Project coordinates onto direction
        proj = coords @ direction

        # Compute 1D Wasserstein distance along projection
        d = wasserstein_distance(proj, proj, P_flat, Q_flat)
        distances.append(d)

    return float(np.mean(distances))


def wasserstein_eval_sliced(
    model: torch.nn.Module,
    test_features: torch.Tensor,
    test_thetas: torch.Tensor,
    logm_range: Tuple[float, float],
    logcp_range: Tuple[float, float],
    ppg,  # PoissonPosteriorGrid
    posteriorbins: int = 100,
    device: str = "cpu",
    n_samples: int = 200,
    n_projections: int = 50,
    seed: Optional[int] = None,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute sliced Wasserstein distance between NN and analytical posteriors.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model.
    test_features : torch.Tensor
        Test observations.
    test_thetas : torch.Tensor
        True parameters.
    logm_range : Tuple[float, float]
        (min, max) for log10(mass).
    logcp_range : Tuple[float, float]
        (min, max) for log10(coupling).
    ppg : PoissonPosteriorGrid
        Analytical posterior grid.
    posteriorbins : int
        Posterior grid resolution.
    device : str
        Device for computation.
    n_samples : int
        Number of samples to evaluate.
    n_projections : int
        Number of random 1D projections for sliced Wasserstein.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    mean_wass : float
        Mean sliced Wasserstein distance.
    results : Dict[str, np.ndarray]
        Dictionary with per-sample results.
    """
    n_total = min(n_samples, len(test_features))
    wass_values, logm_vals, logcp_vals = [], [], []

    for i in range(n_total):
        counts = test_features[i].to(device)
        logm_true, logcp_true = test_thetas[i].cpu().numpy()

        # Analytical posterior
        counts_np = counts.cpu().numpy()
        counts_analytical = counts_np[:100]
        posterior_true = ppg.posterior_binned(counts_analytical)

        # Learned posterior
        counts = preprocess_features(model, counts, device)
        posterior_learned, _, _ = posterior_grid(
            counts, logm_range, logcp_range, posteriorbins, model, device
        )
        posterior_learned = posterior_learned.cpu().numpy()

        # Sliced Wasserstein distance
        wass_val = sliced_wasserstein_2D(
            posterior_true, posterior_learned,
            n_projections=n_projections,
            seed=seed
        )
        
        wass_values.append(wass_val)
        logm_vals.append(logm_true)
        logcp_vals.append(logcp_true)

    mean_wass = float(np.mean(wass_values))
    results = {
        "wass": np.array(wass_values),
        "logm": np.array(logm_vals),
        "logcp": np.array(logcp_vals),
    }
    
    return mean_wass, results


def plot_wasserstein_diagnostics(results, bins=15, cmap="viridis", fill_holes=False, ):
    """
    Visualize Wasserstein values across parameter space.

    Parameters
    ----------
    results : dict
        Output of wasserstein_eval (contains "wass", "logm", "logcp").
    bins : int
        Number of bins per axis for heatmap.
    cmap : str
        Colormap for plots.
    """

    logm = results["logm"]
    logcp = results["logcp"]
    wass = results["wass"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1. Raw scatter ---
    sc = axes[0].scatter(logm, logcp, c=wass, cmap=cmap, s=20)
    axes[0].set_title("Scatter: Per-sample Wasserstein")
    axes[0].set_xlabel(r"log10(mass)")
    axes[0].set_ylabel(r"log10(coupling)")
    plt.colorbar(sc, ax=axes[0], label="Wasserstein Distance")

    # --- 2. Heatmap (binned average) ---
    if fill_holes:
        heatmap, xedges, yedges = compute_filled_heatmap(logm, logcp, wass, bins=bins)

    else:
        heatmap, xedges, yedges, binnum = stats.binned_statistic_2d(
            logm, logcp, wass, statistic="mean", bins=bins
        )

    im = axes[1].imshow(
        heatmap.T, origin="lower", aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap
    )
    axes[1].set_title("Heatmap: Avg. Wasserstein per bin")
    axes[1].set_xlabel("log10(mass)")
    axes[1].set_ylabel("log10(coupling)")
    plt.colorbar(im, ax=axes[1], label="Mean Wasserstein")

    plt.tight_layout(); plt.show()










