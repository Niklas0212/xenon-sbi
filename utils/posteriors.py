"""
Posterior Inference Utilities for WIMP Parameter Estimation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
from torch import nn
from typing import Tuple, Optional, List, Dict
from torch.utils.data import DataLoader

from data.generate_dataset import generate_features
from configs.config import MCConfig, PARAM_RANGES
from utils.processing import preprocess_features, get_matching_pairs


# Visualization settings
TRUE_POINT_COLOR = "#FF3333"  # Red for true parameter markers
M_PROTON_GEV = 0.938
GEV2_TO_CM2 = 3.89379e-28
DEFAULT_XENONNT_PATH = "data\\datasets\\xenon\\s1s2\\official\\xenonnt_2025_si_wimp.csv"

# Shared plotting style for publication-ready readability
PLOT_AXIS_LABEL_SIZE = 16
PLOT_TICK_LABEL_SIZE = 12
PLOT_COLORBAR_LABEL_SIZE = 13
PLOT_COLORBAR_TICK_SIZE = 11
PLOT_LEGEND_SIZE = 11
PLOT_TRUE_POINT_SIZE = 45
PLOT_GRID_LINEWIDTH = 0.8


def _contourf_pdf_safe(ax, x_plot, y_plot, post_plot, levels, cmap):
    """Draw contourf while avoiding PDF seam artifacts across matplotlib versions."""
    im = ax.contourf(x_plot, y_plot, post_plot, levels=levels, cmap=cmap, antialiased=False)
    if hasattr(im, "collections"):  # matplotlib < 3.8
        for col in im.collections:
            col.set_edgecolor("face")
            col.set_linewidth(0.0)
            col.set_antialiased(False)
            col.set_rasterized(False)
    else:  # matplotlib >= 3.8
        im.set_edgecolor("face")
    return im


def _colorbar_pdf_safe(cbar):
    """Style colorbar solids to avoid white seam artifacts in PDF outputs."""
    if hasattr(cbar, "solids") and cbar.solids is not None:
        # Keep colorbar vectorized and overlap polygon boundaries to avoid
        # seam artifacts in PDF viewers.
        cbar.solids.set_edgecolor("face")
        cbar.solids.set_linewidth(0.0)
        cbar.solids.set_antialiased(False)
        cbar.solids.set_rasterized(False)
    if hasattr(cbar, "outline") and cbar.outline is not None:
        cbar.outline.set_visible(False)
    if hasattr(cbar, "dividers") and cbar.dividers is not None:
        cbar.dividers.set_visible(False)
    cbar.drawedges = False
    return cbar

# Default parameter ranges (same as "low" in PARAM_RANGES)
DEFAULT_LOGM_RANGE = PARAM_RANGES["low"]["logm_range"]    # (0.0, 3.0)
DEFAULT_LOGCP_RANGE = PARAM_RANGES["low"]["logcp_range"]   # (-10.5, -8.5)


def _normalize_y_quantity(y_quantity: str) -> str:
    key = str(y_quantity).strip().lower()
    if key in {"cp", "coupling"}:
        return "cp"
    if key in {"sigma", "sigma_p", "cross_section", "cross-section"}:
        return "sigma"
    raise ValueError("y_quantity must be one of {'cp', 'sigma'}")


def _cp_to_sigma_p(m_chi: np.ndarray, cp: np.ndarray) -> np.ndarray:
    m_chi = np.asarray(m_chi, dtype=np.float64)
    cp = np.asarray(cp, dtype=np.float64)
    mu_p = (m_chi * M_PROTON_GEV) / (m_chi + M_PROTON_GEV)
    k_factor = (mu_p**2 / np.pi) * GEV2_TO_CM2
    return k_factor * (cp**2)


def _sigma_p_to_cp(m_chi: np.ndarray, sigma_p: np.ndarray) -> np.ndarray:
    m_chi = np.asarray(m_chi, dtype=np.float64)
    sigma_p = np.asarray(sigma_p, dtype=np.float64)
    mu_p = (m_chi * M_PROTON_GEV) / (m_chi + M_PROTON_GEV)
    k_factor = (mu_p**2 / np.pi) * GEV2_TO_CM2
    return np.sqrt(sigma_p / k_factor)


def _get_ylabel(y_quantity: str) -> str:
    y_quantity = _normalize_y_quantity(y_quantity)
    if y_quantity == "sigma":
        return r"$\sigma_p$ [cm$^2$]"
    return r"$c_p$ [GeV$^{-2}$]"


def _prepare_y_axis(
    m_vals: np.ndarray,
    cp_vals: np.ndarray,
    posterior_np: np.ndarray,
    y_quantity: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform posterior grid arrays for plotting in cp or sigma space.

    Returns (x_plot, y_plot, posterior_plot) ready for contourf.
    Applies the Jacobian correction when converting to sigma.
    """
    y_quantity = _normalize_y_quantity(y_quantity)
    if y_quantity == "cp":
        return m_vals, cp_vals, posterior_np
    # sigma space: need meshgrid for mass-dependent transformation
    M, CP = np.meshgrid(m_vals, cp_vals, indexing="xy")
    Y = _cp_to_sigma_p(M, CP)
    # Jacobian: p(logm, log sigma) = 0.5 * p(logm, log cp)
    return M, Y, posterior_np * 0.5


def _transform_y_value(mchi: float, cp: float, y_quantity: str) -> float:
    """Transform a single (mchi, cp) point to the chosen y-quantity."""
    y_quantity = _normalize_y_quantity(y_quantity)
    if y_quantity == "cp":
        return cp
    return float(_cp_to_sigma_p(mchi, cp))


def load_official_xenonnt_limit(
    official_xenonnt_path: str = DEFAULT_XENONNT_PATH,
    official_max_mass: float = 1000.0,
) -> Optional[Dict[str, np.ndarray]]:
    """Load official XENONnT SI WIMP exclusion line (mass, sigma, cp)."""
    if not os.path.exists(official_xenonnt_path):
        return None

    table = np.genfromtxt(official_xenonnt_path, delimiter=",", names=True)
    if table is None or np.size(table) == 0:
        return None

    mass = np.asarray(table["mass"], dtype=np.float64)
    #sigma = np.asarray(table["upper_limit"], dtype=np.float64) # measured upper limit (not sensitivity)
    sigma = np.asarray(table["sensitivity_0"], dtype=np.float64) # expected sensitivity (better for comparison)
    mask = np.isfinite(mass) & np.isfinite(sigma) & (mass > 0) & (sigma > 0)
    mask &= mass <= official_max_mass
    if not np.any(mask):
        return None

    mass = mass[mask]
    sigma = sigma[mask]
    cp = _sigma_p_to_cp(mass, sigma)
    return {"mass": mass, "sigma": sigma, "cp": cp}


def plot_official_xenonnt_limit(
    ax,
    plot_official_xenonnt: bool = False,
    y_quantity: str = "cp",
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    official_xenonnt_path: str = DEFAULT_XENONNT_PATH,
    official_max_mass: float = 1000.0,
    label: str = "XENONnT 2025 (expected)",
    color: str = TRUE_POINT_COLOR,
    linewidth: float = 2.6,
):
    """Optionally overlay official XENONnT exclusion line on an axis."""
    if not plot_official_xenonnt:
        return None

    y_quantity = _normalize_y_quantity(y_quantity)
    curve = official_xenonnt_limit
    if curve is None:
        curve = load_official_xenonnt_limit(
            official_xenonnt_path=official_xenonnt_path,
            official_max_mass=official_max_mass,
        )
    if curve is None:
        return None

    y_key = "cp" if y_quantity == "cp" else "sigma"
    line, = ax.plot(curve["mass"], curve[y_key], color=color, lw=linewidth, label=label)
    return line


# ==============================================================================
# 1. CORE COMPUTATION
# ==============================================================================

def posterior_grid(
    counts: torch.Tensor,
    logm_range: Tuple[float, float],
    logcp_range: Tuple[float, float],
    posteriorbins: int = 200,
    model: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Compute p(mχ, cp | data) on a 2D logarithmic grid using LRE."""
    assert model is not None, "Model required"

    model.eval()
    with torch.no_grad():
        logm_vals = np.linspace(*logm_range, posteriorbins)
        logcp_vals = np.linspace(*logcp_range, posteriorbins)

        grid_cp, grid_m = np.meshgrid(logcp_vals, logm_vals, indexing="ij")
        theta = np.stack([grid_m, grid_cp], axis=-1).reshape(-1, 2)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=device)

        counts = counts.to(device).unsqueeze(0).expand(len(theta_t), -1)

        score = model(counts, theta_t).squeeze().clamp(1e-10, 1 - 1e-10)
        ler = score / (1 - score)
        posterior = ler

        Z = posterior.sum() * (logm_vals[1]-logm_vals[0]) * (logcp_vals[1]-logcp_vals[0])
        posterior = (posterior / Z).reshape(posteriorbins, posteriorbins)

        return posterior, logm_vals, logcp_vals



# ==============================================================================
# 2. HPD CONTOURS (CREDIBLE REGIONS)
# ==============================================================================
# Highest Posterior Density regions define credible intervals/regions.
# These functions compute and visualize HPD boundaries at specified confidence levels.

def compute_hpd_contours(
    posterior: np.ndarray,
    logm_vals: np.ndarray,
    logcp_vals: np.ndarray,
    cl: float = 0.9,
) -> Tuple[float, List[np.ndarray]]:
    """Compute HPD boundary contours for a given credible level."""
    dcp = logcp_vals[1] - logcp_vals[0]
    dm = logm_vals[1] - logm_vals[0]

    posterior = np.asarray(posterior, dtype=np.float64)
    norm = posterior.sum() * dm * dcp + 1e-12
    post_norm = posterior / norm

    flat = post_norm.ravel()
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order]) * dm * dcp
    idx_thr = np.searchsorted(cum, cl, side="left")
    if idx_thr >= flat.size:
        threshold = flat[order[-1]]
    else:
        threshold = flat[order[idx_thr]]

    fig, ax = plt.subplots(figsize=(6, 4))
    cs = ax.contour(logm_vals, logcp_vals, post_norm, levels=[threshold])
    plt.close(fig)

    contours = []
    if cs.allsegs and len(cs.allsegs[0]) > 0:
        for seg in cs.allsegs[0]:
            if seg.shape[0] >= 2:
                contours.append(seg)

    return float(threshold), contours


def plot_hpd_contours(
    posterior: np.ndarray,
    logm_vals: np.ndarray,
    logcp_vals: np.ndarray,
    cl: float = 0.9,
    ax=None,
    label: str = None,
    color: str = (0.7, 0.7, 0.7),
    linestyle: str = "-",
    linewidth: float = 2.8,
    y_quantity: str = "cp",
):
    """Plot HPD boundary contours."""
    y_quantity = _normalize_y_quantity(y_quantity)
    threshold, contours = compute_hpd_contours(
        posterior, logm_vals, logcp_vals, cl=cl
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for idx, seg in enumerate(contours):
        m_vals = 10 ** seg[:, 0]
        cp_vals = 10 ** seg[:, 1]
        y_vals = cp_vals if y_quantity == "cp" else _cp_to_sigma_p(m_vals, cp_vals)
        ax.plot(
            m_vals,
            y_vals,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if idx == 0 else None,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel(_get_ylabel(y_quantity), fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", which="both", labelsize=PLOT_TICK_LABEL_SIZE)
    #ax.grid(True, which="both", ls="--", lw=PLOT_GRID_LINEWIDTH, alpha=0.5)

    return ax, threshold, contours


def plot_hpd_contours_multi(
    posterior: np.ndarray,
    logm_vals: np.ndarray,
    logcp_vals: np.ndarray,
    credible_levels: List[float] = None,
    ax=None,
    colors: List[str] = None,
    linestyles: List[str] = None,
    linewidths: List[float] = None,
    plot_official_xenonnt: bool = False,
    y_quantity: str = "cp",
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    official_xenonnt_path: str = DEFAULT_XENONNT_PATH,
    official_max_mass: float = 1000.0,
):
    """Plot multiple HPD contours at different credible levels."""
    y_quantity = _normalize_y_quantity(y_quantity)
    if credible_levels is None or len(credible_levels) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        plot_official_xenonnt_limit(
            ax,
            plot_official_xenonnt=plot_official_xenonnt,
            y_quantity=y_quantity,
            official_xenonnt_limit=official_xenonnt_limit,
            official_xenonnt_path=official_xenonnt_path,
            official_max_mass=official_max_mass,
        )
        return ax

    if isinstance(credible_levels, float):
        credible_levels = [credible_levels]

    if colors is None:
        colors = [(0.75, 0.75, 0.75)] * len(credible_levels)

    if linestyles is None:
        if len(credible_levels) <= 1:
            linestyles = ['--'] * len(credible_levels)
        else:
            # inner (tighter) contours dashed, outer (wider) contour solid
            linestyles = ['--'] * (len(credible_levels) - 1) + ['-']
    if linewidths is None:
        linewidths = [2.2] * len(credible_levels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (cl, color, linestyle, linewidth) in enumerate(
        zip(credible_levels, colors, linestyles, linewidths)
    ):
        plot_hpd_contours(
            posterior,
            logm_vals,
            logcp_vals,
            cl=cl,
            ax=ax,
            label=f"{int(cl*100)}% CL exclusion",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            y_quantity=y_quantity,
        )

    plot_official_xenonnt_limit(
        ax,
        plot_official_xenonnt=plot_official_xenonnt,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
        official_xenonnt_path=official_xenonnt_path,
        official_max_mass=official_max_mass,
    )

    return ax



# ==============================================================================
# 3. POSTERIOR VISUALIZATION - WIMPY
# ==============================================================================
# Visualize individual posteriors for specific parameter points.

def plot_posterior_single(
    model: nn.Module,
    mchi: float,
    cp: float,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    mc_config: MCConfig = None,
    top_k: int = 10,
    device: str = "cpu",
    posteriorbins: int = 200,
    levels: int = 50,
    credible_levels: Optional[List[float]] = None,
    plot_title: str = "Posterior",
    cmap: str = "viridis",
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
):
    """Generate synthetic spectrum and plot posterior contours.

    Parameters
    ----------
    model : nn.Module
        Trained SBI classifier.
    mchi : float
        WIMP mass [GeV] for which to generate the spectrum.
    cp : float
        Coupling strength [GeV^-2] for which to generate the spectrum.
    logm_range : tuple, optional
        (logm_min, logm_max). Defaults to PARAM_RANGES["low"].
    logcp_range : tuple, optional
        (logcp_min, logcp_max). Defaults to PARAM_RANGES["low"].
    mc_config : MCConfig, optional
        Monte Carlo configuration (default: MCConfig()).
    top_k : int
        Number of top events to include (default: 10).
    device : str
        Device for computation (default: 'cpu').
    posteriorbins : int
        Grid resolution (default: 200).
    levels : int
        Contour levels (default: 50).
    credible_levels : list of float, optional
        HPD credible levels to overlay (e.g. [0.68, 0.95]).
    plot_title : str
        Plot title (default: "Posterior").
    cmap : str
        Colormap (default: 'viridis').
    y_quantity : str
        Y-axis quantity: 'cp' or 'sigma' (default: 'cp').
    plot_official_xenonnt : bool
        Overlay official XENONnT exclusion line (default: False).
    official_xenonnt_limit : dict, optional
        Pre-loaded XENONnT limit (avoids repeated file I/O).
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    if mc_config is None:
        mc_config = MCConfig()
    y_quantity = _normalize_y_quantity(y_quantity)

    counts, _ = generate_features(mchi, cp, top_k=top_k, **mc_config.__dict__)
    nevents = int(counts[mc_config.bins])
    x = preprocess_features(model, counts, device)

    post, lm, lc = posterior_grid(x, logm_range, logcp_range, posteriorbins, model, device)
    post_np = post.detach().cpu().numpy()

    x_plot, y_plot, post_plot = _prepare_y_axis(10**lm, 10**lc, post_np, y_quantity)
    y_true = _transform_y_value(mchi, cp, y_quantity)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = _contourf_pdf_safe(ax, x_plot, y_plot, post_plot, levels=levels, cmap=cmap)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(
        mchi,
        y_true,
        color=TRUE_POINT_COLOR,
        s=PLOT_TRUE_POINT_SIZE,
        label="True parameters",
        zorder=10,
        alpha=0.9,
    )

    textstr = f"N = {nevents} events"
    legend_alpha = plt.rcParams.get("legend.framealpha", 0.8)
    legend_face = ax.get_facecolor()
    legend_edge = ax.spines["left"].get_edgecolor()
    ax.text(
        0.05,
        0.06,
        textstr,
        transform=ax.transAxes,
        fontsize=PLOT_TICK_LABEL_SIZE,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor=legend_face,
            edgecolor=legend_edge,
            alpha=legend_alpha,
        ),
    )

    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel(_get_ylabel(y_quantity), fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", which="both", labelsize=PLOT_TICK_LABEL_SIZE)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Posterior density", fontsize=PLOT_COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=PLOT_COLORBAR_TICK_SIZE)

    if credible_levels is not None:
        plot_hpd_contours_multi(
            post_np,
            lm,
            lc,
            credible_levels=credible_levels,
            ax=ax,
            y_quantity=y_quantity,
        )

    plot_official_xenonnt_limit(
        ax,
        plot_official_xenonnt=plot_official_xenonnt,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
    )

    ax.legend(loc="lower right", fontsize=PLOT_LEGEND_SIZE)
    plt.tight_layout()
    plt.show()


def plot_posterior_double(
    model: nn.Module,
    mchi: float,
    cp: float,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    mc_config: MCConfig = None,
    top_k: int = 10,
    device: str = "cpu",
    posteriorbins: int = 200,
    levels: int = 50,
    credible_levels: Optional[List[float]] = None,
    plot_title: str = "Posterior",
    cmap: str = "viridis",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """Generate one synthetic spectrum and plot posteriors in cp and sigma panels.

    Left panel uses cp on the y-axis, right panel uses sigma_p.
    Both panels use the same posterior computed from the same spectrum.
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    if mc_config is None:
        mc_config = MCConfig()

    counts, _ = generate_features(mchi, cp, top_k=top_k, **mc_config.__dict__)
    nevents = int(counts[mc_config.bins])
    x = preprocess_features(model, counts, device)

    post, lm, lc = posterior_grid(x, logm_range, logcp_range, posteriorbins, model, device)
    post_np = post.detach().cpu().numpy()

    axis_label_size = PLOT_AXIS_LABEL_SIZE + 3
    tick_label_size = PLOT_TICK_LABEL_SIZE + 3
    colorbar_label_size = PLOT_COLORBAR_LABEL_SIZE + 3
    colorbar_tick_size = PLOT_COLORBAR_TICK_SIZE + 2
    legend_size = PLOT_LEGEND_SIZE + 2
    event_box_font_size = PLOT_TICK_LABEL_SIZE + 3

    x_cp, y_cp, post_cp = _prepare_y_axis(10**lm, 10**lc, post_np, "cp")
    x_sigma, y_sigma, post_sigma = _prepare_y_axis(10**lm, 10**lc, post_np, "sigma")

    y_true_cp = _transform_y_value(mchi, cp, "cp")
    y_true_sigma = _transform_y_value(mchi, cp, "sigma")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    panel_data = [
        (axes[0], x_cp, y_cp, post_cp, y_true_cp, "cp"),
        (axes[1], x_sigma, y_sigma, post_sigma, y_true_sigma, "sigma"),
    ]

    for idx, (ax, x_plot, y_plot, post_plot, y_true, y_quantity) in enumerate(panel_data):
        if np.isscalar(levels):
            panel_vmax = float(np.nanmax(post_plot))
            panel_levels = np.linspace(0.0, panel_vmax, int(levels))
        else:
            panel_levels = levels
        im = _contourf_pdf_safe(ax, x_plot, y_plot, post_plot, levels=panel_levels, cmap=cmap)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(
            mchi,
            y_true,
            color=TRUE_POINT_COLOR,
            s=PLOT_TRUE_POINT_SIZE,
            label="True parameters" if idx == 1 else None,
            zorder=10,
            alpha=0.9,
        )

        if idx == 0:
            textstr = f"N = {nevents} events"
            legend_alpha = plt.rcParams.get("legend.framealpha", 0.8)
            legend_face = ax.get_facecolor()
            legend_edge = ax.spines["left"].get_edgecolor()
            ax.text(
                0.05,
                0.10,
                textstr,
                transform=ax.transAxes,
                fontsize=event_box_font_size,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor=legend_face,
                    edgecolor=legend_edge,
                    alpha=legend_alpha,
                ),
            )

        ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_size)
        ax.set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_size)
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)

        if credible_levels is not None:
            plot_hpd_contours_multi(
                post_np,
                lm,
                lc,
                credible_levels=credible_levels,
                ax=ax,
                y_quantity=y_quantity,
            )

        plot_official_xenonnt_limit(
            ax,
            plot_official_xenonnt=plot_official_xenonnt,
            y_quantity=y_quantity,
            official_xenonnt_limit=official_xenonnt_limit,
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        if idx == 1:
            cbar.set_label("Posterior density", fontsize=colorbar_label_size, labelpad=10)
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.formatter = ticker.FuncFormatter(lambda x, pos: f"{x:#.2g}")
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=colorbar_tick_size)
        if idx == 1:
            ax.legend(loc="lower right", fontsize=legend_size)

    fig.subplots_adjust(wspace=0.46, top=0.88)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()


def plot_posterior_grid(
    model: nn.Module,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    mc_config: MCConfig = None,
    top_k: int = 10,
    device: str = "cpu",
    posteriorbins: int = 200,
    grid_dim: int = 4,
    from_validation: bool = False,
    val_loader: Optional[DataLoader] = None,
    stochastic: bool = True,
    cmap: str = "viridis",
    colorbar: bool = False,
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """Plot a grid of posteriors (synthetic or from validation).

    Parameters
    ----------
    model : nn.Module
        Trained SBI classifier.
    logm_range / logcp_range : tuple, optional
        Parameter ranges (default: PARAM_RANGES["low"]).
    mc_config : MCConfig, optional
        Monte Carlo configuration (default: MCConfig()).
    top_k : int
        Number of top events (default: 10).
    device : str
        Device for computation (default: 'cpu').
    from_validation : bool
        If True, pick spectra from *val_loader* instead of generating.
    val_loader : DataLoader, optional
        Required when *from_validation* is True.
    y_quantity : str
        Y-axis quantity: 'cp' or 'sigma' (default: 'cp').
    save_path : str, optional
        If provided, save the figure to this path.
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    if mc_config is None:
        mc_config = MCConfig()
    y_quantity = _normalize_y_quantity(y_quantity)

    # Larger typography for improved readability in multi-panel figures.
    axis_label_size = PLOT_AXIS_LABEL_SIZE - 4
    tick_label_size = PLOT_TICK_LABEL_SIZE - 1
    event_box_font_size = PLOT_TICK_LABEL_SIZE 

    def _decode_nevents_scalar(n_val: float) -> int:
        """Decode N from either raw count or log10(N+1)-transformed scalar."""
        if np.isfinite(n_val) and np.isclose(n_val, np.round(n_val), atol=1e-6):
            return int(max(0, np.round(n_val)))
        return int(max(0, np.round((10.0 ** n_val) - 1.0)))

    logms = np.linspace(*logm_range, grid_dim+2)[1:-1]
    logcps = np.linspace(*logcp_range, grid_dim+2)[1:-1]
    pairs = [(10**m, 10**c) for c in logcps for m in logms]

    if from_validation:
        assert val_loader, "Need val_loader for from_validation=True"
        val_x, val_t = get_matching_pairs("val", None, val_loader, None)

    fig, axes = plt.subplots(
        grid_dim, grid_dim,
        figsize=(2.0 * grid_dim + 0.5, 2.0 * grid_dim + 0.5),
        sharex=True, sharey=True
    )
    axes = axes.reshape(grid_dim, grid_dim)
    ims = []
    official_curve = official_xenonnt_limit
    if plot_official_xenonnt and official_curve is None:
        official_curve = load_official_xenonnt_limit()

    for idx, (mchi, cp) in enumerate(pairs):
        r, c = divmod(idx, grid_dim)
        ax = axes[r, c]

        if from_validation:
            dist = (val_t[:, 0] - np.log10(mchi))**2 + (val_t[:, 1] - np.log10(cp))**2
            k = 10 if stochastic else 1
            nearest = np.argsort(dist)[:k]
            pick = np.random.choice(nearest)
            x = val_x[pick:pick+1].squeeze(0).to(device)
            x_flat = x.detach().reshape(-1)
            model_name = model.__class__.__name__
            if x_flat.numel() > mc_config.bins:
                nevents = _decode_nevents_scalar(float(x_flat[mc_config.bins].item()))
            elif model_name in {"Ntot_MLP", "Ntot_Highest_MLP", "Ntot_Highest_MLP_Vanilla"} and x_flat.numel() >= 1:
                nevents = _decode_nevents_scalar(float(x_flat[0].item()))
            else:
                # Fallback for feature sets without explicit N_tot.
                # If features look log-transformed (WimPy models), invert approximately per bin.
                x_np = x_flat.detach().cpu().numpy()
                if np.all(x_np >= 0.0) and np.nanmax(x_np) <= 6.0:
                    nevents = int(max(0, np.round(np.sum((10.0 ** x_np) - 1.0))))
                else:
                    nevents = int(max(0, np.round(np.sum(x_np))))
        else:
            counts, _ = generate_features(mchi, cp, top_k=top_k, **mc_config.__dict__)
            nevents = int(counts[mc_config.bins])
            x = preprocess_features(model, counts, device)

        post, lm, lc = posterior_grid(x, logm_range, logcp_range, posteriorbins, model, device)
        post_np = post.detach().cpu().numpy()

        x_plot, y_plot, post_plot = _prepare_y_axis(10**lm, 10**lc, post_np, y_quantity)
        y_true = _transform_y_value(mchi, cp, y_quantity)

        im = _contourf_pdf_safe(ax, x_plot, y_plot, post_plot, levels=40, cmap=cmap)
        ims.append(im)
        ax.scatter(mchi, y_true, color=TRUE_POINT_COLOR, s=22)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plot_official_xenonnt_limit(
            ax,
            plot_official_xenonnt=plot_official_xenonnt,
            y_quantity=y_quantity,
            official_xenonnt_limit=official_curve,
        )
        ax.text(
            0.94,
            0.06,
            f"N={nevents}",
            transform=ax.transAxes,
            fontsize=event_box_font_size,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
        )

    for ax in axes[-1]:
        ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_size)
    for ax in axes[:, 0]:
        ax.set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_size)

    for ax in axes.flat:
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)

    if colorbar:
        for ax, im in zip(axes.flat, ims):
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=tick_label_size)
            
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()


def plot_null_exclusion(
    model: nn.Module,
    feature_template: torch.Tensor,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    device: str = "cpu",
    posteriorbins: int = 500,
    credible_levels: Optional[List[float]] = None,
    levels: int = 200,
    cmap: str = "viridis",
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    plot_title: str = "Null Posterior with Upper Limit Exclusion",
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
):
    """Compute and plot the null-spectrum posterior with HPD exclusion contours.

    Parameters
    ----------
    model : nn.Module
        Trained SBI classifier.
    feature_template : torch.Tensor
        Any feature tensor whose shape is used to create the null (zero) spectrum.
    logm_range / logcp_range : tuple, optional
        Parameter ranges (default: PARAM_RANGES["low"]).
    device : str
        Device for computation (default: 'cpu').
    posteriorbins : int
        Grid resolution (default: 500).
    credible_levels : list of float, optional
        HPD credible levels to overlay (default: [0.90]).
    levels : int
        Contour fill levels (default: 200).
    y_quantity : str
        Y-axis quantity: 'cp' or 'sigma' (default: 'cp').
    plot_official_xenonnt : bool
        Overlay official XENONnT exclusion line (default: False).
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    if credible_levels is None:
        credible_levels = [0.90]
    y_quantity = _normalize_y_quantity(y_quantity)

    # Null spectrum (all zeros)
    null_counts = torch.zeros_like(feature_template)

    posterior_null, logm_vals, logcp_vals = posterior_grid(
        null_counts, logm_range, logcp_range, posteriorbins, model, device
    )
    posterior_null_np = posterior_null.detach().cpu().numpy()

    axis_label_size = PLOT_AXIS_LABEL_SIZE + 4
    tick_label_size = PLOT_TICK_LABEL_SIZE + 4
    colorbar_label_size = PLOT_COLORBAR_LABEL_SIZE + 4
    colorbar_tick_size = PLOT_COLORBAR_TICK_SIZE + 3
    legend_size = PLOT_LEGEND_SIZE + 3

    x_plot, y_plot, post_plot = _prepare_y_axis(
        10**logm_vals, 10**logcp_vals, posterior_null_np, y_quantity
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = _contourf_pdf_safe(ax, x_plot, y_plot, post_plot, levels=levels, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    _colorbar_pdf_safe(cbar)
    cbar.set_label("Posterior density", fontsize=colorbar_label_size, labelpad=10)
    cbar.locator = ticker.MaxNLocator(nbins=4)
    cbar.formatter = ticker.FuncFormatter(lambda x, pos: f"{x:#.2g}")
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=colorbar_tick_size)

    if isinstance(credible_levels, float):
        credible_levels_plot = [credible_levels]
    else:
        credible_levels_plot = list(credible_levels)
    contour_colors = [TRUE_POINT_COLOR] * len(credible_levels_plot)
    contour_linestyles = ["-"] * len(credible_levels_plot)
    contour_linewidths = [3.2] * len(credible_levels_plot)

    plot_hpd_contours_multi(
        posterior_null_np,
        logm_vals,
        logcp_vals,
        credible_levels=credible_levels_plot,
        ax=ax,
        colors=contour_colors,
        linestyles=contour_linestyles,
        linewidths=contour_linewidths,
        plot_official_xenonnt=False,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
    )

    plot_official_xenonnt_limit(
        ax,
        plot_official_xenonnt=plot_official_xenonnt,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
        color=(0.5, 0.5, 0.5),
        linewidth=3.2,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_size)
    ax.set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_size)
    ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=legend_size)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()



# ==============================================================================
# 4. POSTERIOR VISUALIZATION - S1S2 SPECIFIC
# ==============================================================================
# Specialized plotting functions for S1S2 histogram data with validation datasets.

def plot_posterior_single_s1s2(
    model: nn.Module,
    val_loader: DataLoader,
    mchi: float,
    cp: float,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    device: str = "cpu",
    posteriorbins: int = 200,
    levels: int = 50,
    credible_levels: Optional[List[float]] = None,
    plot_title: str = r"Posterior ($S_1S_2$)",
    cmap: str = "viridis",
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """Plot posterior for S1S2 histogram closest to specified parameters.

    Parameters
    ----------
    model : nn.Module
        Trained SBI classifier.
    val_loader : DataLoader
        Validation data loader with S1S2 histograms.
    mchi : float
        WIMP mass [GeV].
    cp : float
        Coupling strength [GeV^-2].
    logm_range / logcp_range : tuple, optional
        Parameter ranges (default: PARAM_RANGES["low"]).
    device : str
        Device for computation (default: 'cpu').
    y_quantity : str
        Y-axis quantity: 'cp' or 'sigma' (default: 'cp').
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    y_quantity = _normalize_y_quantity(y_quantity)

    val_features, val_thetas = get_matching_pairs("val", None, val_loader, None)

    log_m_true, log_cp_true = np.log10(mchi), np.log10(cp)
    val_thetas_np = val_thetas.cpu().numpy()
    dist = (val_thetas_np[:, 0] - log_m_true) ** 2 + (val_thetas_np[:, 1] - log_cp_true) ** 2

    idx = np.argmin(dist)
    counts = val_features[idx:idx+1].flatten().to(device)

    post, logm_vals, logcp_vals = posterior_grid(counts, logm_range, logcp_range, posteriorbins, model, device)
    post_np = post.detach().cpu().numpy()

    x_plot, y_plot, post_plot = _prepare_y_axis(10**logm_vals, 10**logcp_vals, post_np, y_quantity)
    y_true = _transform_y_value(mchi, cp, y_quantity)

    nevents = int(counts.sum().item())

    fig, ax = plt.subplots(figsize=(6, 5))
    im = _contourf_pdf_safe(ax, x_plot, y_plot, post_plot, levels=levels, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    _colorbar_pdf_safe(cbar)
    cbar.set_label("Posterior density", fontsize=PLOT_COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=PLOT_COLORBAR_TICK_SIZE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel(_get_ylabel(y_quantity), fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", which="both", labelsize=PLOT_TICK_LABEL_SIZE)

    ax.scatter(mchi, y_true, color=TRUE_POINT_COLOR, s=PLOT_TRUE_POINT_SIZE, label="True parameters", zorder=10)

    textstr = f"N = {nevents} events"
    legend_alpha = plt.rcParams.get("legend.framealpha", 0.8)
    legend_face = ax.get_facecolor()
    legend_edge = ax.spines["left"].get_edgecolor()
    ax.text(
        0.05,
        0.06,
        textstr,
        transform=ax.transAxes,
        fontsize=PLOT_TICK_LABEL_SIZE,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor=legend_face,
            edgecolor=legend_edge,
            alpha=legend_alpha,
        ),
    )

    if credible_levels is not None:
        plot_hpd_contours_multi(
            post_plot,
            logm_vals,
            logcp_vals,
            credible_levels=credible_levels,
            ax=ax,
            y_quantity=y_quantity,
        )
    plot_official_xenonnt_limit(
        ax,
        plot_official_xenonnt=plot_official_xenonnt,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
    )
    ax.legend(loc="lower right", fontsize=PLOT_LEGEND_SIZE)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()


def plot_posterior_grid_s1s2(
    model: nn.Module,
    val_loader: DataLoader,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    device: str = "cpu",
    posteriorbins: int = 200,
    grid_dim: int = 4,
    from_validation: bool = True,
    stochastic: bool = True,
    cmap: str = "viridis",
    colorbar: bool = False,
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """Plot a grid of posteriors from S1S2 validation dataset.

    Parameters
    ----------
    model : nn.Module
        Trained SBI classifier.
    val_loader : DataLoader
        Validation data loader with S1S2 histograms.
    logm_range / logcp_range : tuple, optional
        Parameter ranges (default: PARAM_RANGES["low"]).
    device : str
        Device for computation (default: 'cpu').
    posteriorbins : int
        Grid resolution (default: 200).
    grid_dim : int
        Number of rows/columns in the grid (default: 4).
    stochastic : bool
        If True, randomly pick among 10 nearest neighbours (default: True).
    y_quantity : str
        Y-axis quantity: 'cp' or 'sigma' (default: 'cp').
    plot_official_xenonnt : bool
        Overlay official XENONnT exclusion line (default: False).
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    y_quantity = _normalize_y_quantity(y_quantity)
    if not from_validation:
        raise ValueError("plot_posterior_grid_s1s2 requires from_validation=True")

    axis_label_size = PLOT_AXIS_LABEL_SIZE - 4
    tick_label_size = PLOT_TICK_LABEL_SIZE - 1
    event_box_font_size = PLOT_TICK_LABEL_SIZE

    val_features, val_thetas = get_matching_pairs("val", None, val_loader, None)
    val_thetas_np = val_thetas.cpu().numpy()

    logms = np.linspace(*logm_range, grid_dim + 2)[1:-1]
    logcps = np.linspace(*logcp_range, grid_dim + 2)[1:-1]

    fig, axes = plt.subplots(
        grid_dim, grid_dim,
        figsize=(2.0 * grid_dim + 0.5, 2.0 * grid_dim + 0.5),
        sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(grid_dim, grid_dim)
    ims = []
    official_curve = official_xenonnt_limit
    if plot_official_xenonnt and official_curve is None:
        official_curve = load_official_xenonnt_limit()

    for r in range(grid_dim):
        for c in range(grid_dim):
            ax = axes[r, c]
            logm = logms[c]
            logcp = logcps[r]

            dist = (val_thetas_np[:, 0] - logm) ** 2 + (val_thetas_np[:, 1] - logcp) ** 2
            k = 10 if stochastic else 1
            nearest = np.argsort(dist)[:k]
            pick = np.random.choice(nearest)
            counts = val_features[pick:pick+1].flatten().to(device)

            post, logm_vals, logcp_vals = posterior_grid(counts, logm_range, logcp_range, posteriorbins, model, device)
            post_np = post.detach().cpu().numpy()

            x_plot, y_plot, post_plot = _prepare_y_axis(10**logm_vals, 10**logcp_vals, post_np, y_quantity)
            y_true = _transform_y_value(10**logm, 10**logcp, y_quantity)

            im = _contourf_pdf_safe(ax, x_plot, y_plot, post_plot, levels=40, cmap=cmap)
            ims.append(im)

            ax.scatter(10 ** logm, y_true, color=TRUE_POINT_COLOR, s=22)
            ax.set_xscale("log")
            ax.set_yscale("log")
            plot_official_xenonnt_limit(
                ax,
                plot_official_xenonnt=plot_official_xenonnt,
                y_quantity=y_quantity,
                official_xenonnt_limit=official_curve,
            )
            nevents = int(counts.sum().item())
            ax.text(
                0.94,
                0.06,
                f"N={nevents}",
                transform=ax.transAxes,
                fontsize=event_box_font_size,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
            )

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_size)
    for ax in axes[:, 0]:
        ax.set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_size)

    for ax in axes.flat:
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)

    if colorbar:
        for ax, im in zip(axes.flat, ims):
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            _colorbar_pdf_safe(cbar)
            cbar.ax.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()


def generate_null_spectrum_s1s2(
    background_events: np.ndarray,
    mu_bg: float = 0,
    s1_bins: int = 30,
    s2_bins: int = 30,
    s1_range: Tuple[float, float] = (0, 100),
    s2_range: Tuple[float, float] = (10**2.1, 10**4.1),
    rng_seed: int = 42,
) -> torch.Tensor:
    """Generate a null (background-only) spectrum for S1S2 analysis."""
    s1_edges = np.linspace(*s1_range, s1_bins + 1)
    s2_edges = np.logspace(np.log10(s2_range[0]), np.log10(s2_range[1]), s2_bins + 1)

    if mu_bg == 0:
        null_hist = np.zeros((s1_bins, s2_bins), dtype=np.float32)
    else:
        rng = np.random.default_rng(rng_seed)
        N_bg = rng.poisson(mu_bg)

        if N_bg == 0:
            null_hist = np.zeros((s1_bins, s2_bins), dtype=np.float32)
        else:
            n_bg_total = background_events.shape[0]
            idx = rng.integers(0, n_bg_total, size=N_bg)
            bg_sample = background_events[idx]

            cS1, cS2 = bg_sample[:, 0], bg_sample[:, 1]
            null_hist, _, _ = np.histogram2d(cS1, cS2, bins=[s1_edges, s2_edges])

    null_spectrum = torch.tensor(null_hist.flatten(), dtype=torch.float32)
    return null_spectrum


def plot_null_exclusion_s1s2(
    model: nn.Module,
    background_events: np.ndarray,
    mu_bg: float = 0,
    bins: int = 30,
    rng_seed: int = 42,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    device: str = "cpu",
    posteriorbins: int = 500,
    credible_levels: Optional[List[float]] = None,
    levels: int = 200,
    cmap: str = "viridis",
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    plot_title: str = "Null Posterior with Upper Limit Exclusion",
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
):
    """Compute and plot the S1S2 null-spectrum posterior with HPD exclusion contours.

    Parameters
    ----------
    model : nn.Module
        Trained SBI classifier.
    background_events : np.ndarray
        Background event pool (cS1, cS2).
    mu_bg : float
        Mean number of background events (0 = signal-only null).
    bins : int
        Number of histogram bins per S1/S2 axis (default: 30).
    rng_seed : int
        Random seed for background sampling (default: 42).
    logm_range / logcp_range : tuple, optional
        Parameter ranges (default: PARAM_RANGES["low"]).
    device : str
        Device for computation (default: 'cpu').
    posteriorbins : int
        Grid resolution (default: 500).
    credible_levels : list of float, optional
        HPD credible levels to overlay (default: [0.90]).
    levels : int
        Contour fill levels (default: 200).
    y_quantity : str
        Y-axis quantity: 'cp' or 'sigma' (default: 'cp').
    plot_official_xenonnt : bool
        Overlay official XENONnT exclusion line (default: False).
    plot_title : str
        Plot title.
    figsize : tuple
        Figure size (default: (8, 6)).
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    if credible_levels is None:
        credible_levels = [0.90]
    y_quantity = _normalize_y_quantity(y_quantity)

    # Local size parameters for this plot (slightly enlarged for better visibility)
    axis_label_size = 20
    tick_label_size = 17
    colorbar_label_size = 18
    colorbar_tick_size = 15
    legend_size = 15

    null_spectrum = generate_null_spectrum_s1s2(
        background_events=background_events,
        mu_bg=mu_bg,
        s1_bins=bins,
        s2_bins=bins,
        rng_seed=rng_seed,
    )

    posterior_null, logm_vals, logcp_vals = posterior_grid(
        null_spectrum, logm_range, logcp_range, posteriorbins, model, device
    )
    posterior_null_np = posterior_null.detach().cpu().numpy()

    x_plot, y_plot, post_plot = _prepare_y_axis(
        10**logm_vals, 10**logcp_vals, posterior_null_np, y_quantity
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = _contourf_pdf_safe(ax, x_plot, y_plot, post_plot, levels=levels, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    _colorbar_pdf_safe(cbar)
    cbar.set_label("Posterior density", fontsize=colorbar_label_size)
    cbar.locator = ticker.MaxNLocator(nbins=4)
    cbar.formatter = ticker.FuncFormatter(lambda x, pos: f"{x:#.2g}")
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=colorbar_tick_size)

    if isinstance(credible_levels, float):
        credible_levels_plot = [credible_levels]
    else:
        credible_levels_plot = list(credible_levels)
    contour_colors = [TRUE_POINT_COLOR] * len(credible_levels_plot)
    contour_linestyles = ["-"] * len(credible_levels_plot)
    contour_linewidths = [3.2] * len(credible_levels_plot)

    plot_hpd_contours_multi(
        posterior_null_np,
        logm_vals,
        logcp_vals,
        credible_levels=credible_levels_plot,
        ax=ax,
        colors=contour_colors,
        linestyles=contour_linestyles,
        linewidths=contour_linewidths,
        plot_official_xenonnt=False,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
    )

    plot_official_xenonnt_limit(
        ax,
        plot_official_xenonnt=plot_official_xenonnt,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
        color=(0.5, 0.5, 0.5),
        linewidth=3.2,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_size)
    ax.set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_size)
    ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=legend_size)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()


def _s1s2_likelihood_ratio_grid(
    counts: torch.Tensor,
    logm_range: Tuple[float, float],
    logcp_range: Tuple[float, float],
    posteriorbins: int,
    model: nn.Module,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the unnormalized likelihood-ratio grid used by the SBI classifier."""
    model.eval()
    with torch.no_grad():
        logm_vals = np.linspace(*logm_range, posteriorbins)
        logcp_vals = np.linspace(*logcp_range, posteriorbins)

        grid_cp, grid_m = np.meshgrid(logcp_vals, logm_vals, indexing="ij")
        theta = np.stack([grid_m, grid_cp], axis=-1).reshape(-1, 2)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=device)

        counts = counts.to(device).reshape(1, -1).expand(len(theta_t), -1)
        score = model(counts, theta_t).squeeze().clamp(1e-10, 1 - 1e-10)
        lr = (score / (1 - score)).reshape(posteriorbins, posteriorbins)

        return lr.detach().cpu().numpy(), logm_vals, logcp_vals


def compute_profile_likelihood_upper_limit_s1s2(
    background_events: np.ndarray,
    mu_bg: float,
    bins: int = 10,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    model: nn.Module = None,
    device: str = "cpu",
    rng_seed: int = 42,
    posteriorbins: int = 200,
    wilks_q: float = 2.71,
    cl: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 90% CL profile-likelihood-style upper limit for one S1S2 realization."""
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE

    null_spectrum = generate_null_spectrum_s1s2(
        background_events=background_events,
        mu_bg=mu_bg,
        s1_bins=bins,
        s2_bins=bins,
        rng_seed=rng_seed,
    )

    lr_grid, logm_vals, logcp_vals = _s1s2_likelihood_ratio_grid(
        null_spectrum,
        logm_range=logm_range,
        logcp_range=logcp_range,
        posteriorbins=posteriorbins,
        model=model,
        device=device,
    )

    delta = float(np.exp(-wilks_q / 2.0))
    n_cp, n_mass = lr_grid.shape
    logcp_limits = np.full(n_mass, np.nan)

    for m_idx in range(n_mass):
        col = lr_grid[:, m_idx]
        if not np.any(np.isfinite(col)):
            continue

        r_max = np.nanmax(col)
        if not np.isfinite(r_max) or r_max <= 0:
            continue

        prof = col / r_max
        allowed = np.isfinite(prof) & (prof >= delta)

        if not np.any(allowed):
            continue
        peak_idx = int(np.nanargmax(col))
        if not allowed[peak_idx]:
            continue

        upper_idx = peak_idx
        while upper_idx < n_cp - 1 and allowed[upper_idx + 1]:
            upper_idx += 1

        if upper_idx >= n_cp - 1:
            logcp_limits[m_idx] = logcp_vals[-1]
            continue

        left = prof[upper_idx]
        right = prof[upper_idx + 1]
        if not np.isfinite(left) or not np.isfinite(right) or np.isclose(left, right):
            logcp_limits[m_idx] = logcp_vals[upper_idx]
            continue

        frac = (delta - left) / (right - left)
        frac = np.clip(frac, 0.0, 1.0)
        logcp_limits[m_idx] = logcp_vals[upper_idx] + frac * (logcp_vals[upper_idx + 1] - logcp_vals[upper_idx])

    return logcp_limits, logm_vals


def compute_profile_likelihood_limit_statistics(
    background_events: np.ndarray,
    mu_bg: float,
    bins: int = 10,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    model: nn.Module = None,
    device: str = "cpu",
    n_realizations: int = 100,
    seed_offset: int = 0,
    posteriorbins: int = 200,
    wilks_q: float = 2.71,
    cl: float = 0.90,
    smoothing_window: int = 10,
) -> Tuple[dict, np.ndarray, List[np.ndarray]]:
    """Compute profile-likelihood limit statistics across multiple S1S2 background realizations."""
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE

    seeds = np.arange(n_realizations) + seed_offset
    logcp_limits_list = []
    logm_vals = None

    for seed in seeds:
        logcp_lim, logm_vals_iter = compute_profile_likelihood_upper_limit_s1s2(
            background_events=background_events,
            mu_bg=mu_bg,
            bins=bins,
            logm_range=logm_range,
            logcp_range=logcp_range,
            model=model,
            device=device,
            rng_seed=int(seed),
            posteriorbins=posteriorbins,
            wilks_q=wilks_q,
            cl=cl,
        )

        if logm_vals is None:
            logm_vals = logm_vals_iter

        logcp_limits_list.append(logcp_lim)

    stats = compute_exclusion_statistics(
        logcp_limits_list,
        smoothing_window=smoothing_window,
    )

    return stats, logm_vals, logcp_limits_list


def plot_profile_likelihood_limits_with_bands(
    logm_vals: np.ndarray,
    stats: dict,
    show_mean: bool = True,
    show_median: bool = True,
    critical_mass: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 6),
    axis_label_fontsize: float = PLOT_AXIS_LABEL_SIZE,
    tick_label_fontsize: float = PLOT_TICK_LABEL_SIZE,
    legend_fontsize: float = PLOT_LEGEND_SIZE,
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """Plot frequentist profile-likelihood upper limits with uncertainty bands."""
    y_quantity = _normalize_y_quantity(y_quantity)
    m_vals = 10 ** logm_vals

    valid = np.isfinite(stats["median"])
    if critical_mass is not None:
        above_threshold = m_vals >= critical_mass
        valid = valid & above_threshold

    m_vals = m_vals[valid]
    cp_mean = 10 ** stats["mean"][valid]
    cp_med = 10 ** stats["median"][valid]
    cp_1lo = 10 ** stats["p16"][valid]
    cp_1hi = 10 ** stats["p84"][valid]
    cp_2lo = 10 ** stats["p2p5"][valid]
    cp_2hi = 10 ** stats["p97p5"][valid]

    if y_quantity == "sigma":
        y_mean = _cp_to_sigma_p(m_vals, cp_mean)
        y_med = _cp_to_sigma_p(m_vals, cp_med)
        y_1lo = _cp_to_sigma_p(m_vals, cp_1lo)
        y_1hi = _cp_to_sigma_p(m_vals, cp_1hi)
        y_2lo = _cp_to_sigma_p(m_vals, cp_2lo)
        y_2hi = _cp_to_sigma_p(m_vals, cp_2hi)
    else:
        y_mean = cp_mean
        y_med = cp_med
        y_1lo = cp_1lo
        y_1hi = cp_1hi
        y_2lo = cp_2lo
        y_2hi = cp_2hi

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(m_vals, y_1lo, y_1hi, color="green", alpha=0.50, label=r"68% band (1$\sigma$)")
    ax.fill_between(m_vals, y_2lo, y_2hi, color="yellow", alpha=0.50, label=r"95% band (2$\sigma$)")

    if show_mean:
        ax.plot(m_vals, y_mean, color="blue", linewidth=2.0, linestyle="-", label="Mean upper limit")
    if show_median:
        ax.plot(m_vals, y_med, color="black", linewidth=3.2, linestyle="--", label="Median UL (frequentist)")

    plot_official_xenonnt_limit(
        ax,
        plot_official_xenonnt=plot_official_xenonnt,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
        color=(0.5, 0.5, 0.5),
        linewidth=3.2,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    if y_quantity == "sigma":
        m_rep = np.sqrt(1.0 * 1000.0)
        ax.set_ylim(
            _cp_to_sigma_p(m_rep, 10 ** (-10.5)),
            _cp_to_sigma_p(m_rep, 10 ** (-8.5)),
        )
    else:
        ax.set_ylim(10 ** (-10.5), 10 ** (-8.5))
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_fontsize)
    ax.set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=tick_label_fontsize)
    #ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=legend_fontsize)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()


def plot_hpd_contours_multiple_bg_realizations(
    background_events: np.ndarray,
    mu_bg: float,
    bins: int = 30,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    model: nn.Module = None,
    device: str = "cpu",
    n_realizations: int = 10,
    seed_offset: int = 0,
    posteriorbins: int = 200,
    cl: float = 0.90,
    figsize: Tuple[float, float] = (8, 6),
    linewidth: float = 2.2,
    legend_linewidth: float = 5.0,
    legend_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0),
    legend_fontsize: float = 16,
    save_path: Optional[str] = None,
):
    """Plot HPD contours overlaid for multiple background realizations.
    
    Shows how background fluctuations affect the contour locations,
    illustrating why simple contours cannot reliably define upper limits.
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    seeds = np.arange(n_realizations) + seed_offset
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, seed in enumerate(seeds):
        null_spec = generate_null_spectrum_s1s2(
            background_events=background_events,
            mu_bg=mu_bg,
            s1_bins=bins,
            s2_bins=bins,
            rng_seed=int(seed),
        )
        
        posterior, logm_vals, logcp_vals = posterior_grid(
            null_spec,
            logm_range=logm_range,
            logcp_range=logcp_range,
            posteriorbins=posteriorbins,
            model=model,
            device=device,
        )
        
        posterior_np = posterior.detach().cpu().numpy()
        
        plot_hpd_contours(
            posterior_np,
            logm_vals,
            logcp_vals,
            cl=cl,
            ax=ax,
            color=(0.2, 0.2, 0.2, 0.3),
            linestyle='-',
            linewidth=linewidth,
            label=f"{int(cl*100)}% HPD" if i == 0 else None,
        )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.set_ylabel(r"$c_p$ [GeV$^{-2}$]")
    #ax.grid(True, which="both", ls="--", alpha=0.5)
    legend = ax.legend(loc="lower right", fontsize=legend_fontsize)
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    for handle in handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(legend_linewidth)
        if hasattr(handle, "set_color"):
            handle.set_color(legend_color)
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(legend_color[3])
    ax.tick_params(axis='both', which='major', labelsize=legend_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=legend_fontsize)
    ax.xaxis.label.set_size(legend_fontsize)
    ax.yaxis.label.set_size(legend_fontsize)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()



# ==============================================================================
# 5. MARGINALIZED UPPER LIMITS & EXCLUSION
# ==============================================================================
# Core functions for computing 1D marginalized upper limits on coupling parameters.
# Includes critical mass determination, threshold application, and visualization.

def compute_marginalized_upper_limit(
    posterior: np.ndarray,
    logm_vals: np.ndarray,
    logcp_vals: np.ndarray,
    cl: float = 0.90,
) -> np.ndarray:
    """
    Compute 1D marginalized upper limits on coupling at each mass.
    
    For each mass value, marginalizes the 2D posterior over coupling to get p(cp | m, data),
    then finds the coupling value where the cumulative probability reaches the credible level.
    
    Args:
        posterior: 2D posterior grid, shape (n_cp, n_mass)
        logm_vals: log10(mass) grid centers, shape (n_mass,)
        logcp_vals: log10(coupling) grid centers, shape (n_cp,)
        cl: Credible level (default 0.90 for 90% CL upper limit)
    
    Returns:
        logcp_upper_limits: Array of log10(coupling) upper limits at each mass
                           NaN where limit exceeds grid range
    """
    dcp = logcp_vals[1] - logcp_vals[0]  

    posterior = np.asarray(posterior, dtype=np.float64)
    n_cp, n_mass = posterior.shape

    logcp_upper_limits = np.full(n_mass, np.nan)  # Initialize to NaN for invalid cases

    # Process each mass bin independently
    for m_idx in range(n_mass):

        post_1d = posterior[:, m_idx]

        norm = post_1d.sum() * dcp
        if norm < 1e-12:
            continue

        post_1d_norm = post_1d / norm
        cumulative = np.cumsum(post_1d_norm) * dcp
        idx_limit = np.searchsorted(cumulative, cl, side='left')

        # Handle edge cases
        if idx_limit >= n_cp:
            # Posterior peak is beyond grid → limit exceeds parameter range
            logcp_upper_limits[m_idx] = np.nan
        elif idx_limit == 0:
            # Limit is at or below lowest grid point
            logcp_upper_limits[m_idx] = logcp_vals[0]
        else:
            # Valid posterior → interpolate to get accurate limit
            if idx_limit < n_cp - 1:
                # Linear interpolation between grid points for smoother limit
                frac = (cl - cumulative[idx_limit - 1]) / (cumulative[idx_limit] - cumulative[idx_limit - 1] + 1e-12)
                logcp_upper_limits[m_idx] = logcp_vals[idx_limit - 1] + frac * dcp
            else:
                # At the edge of grid; use the edge value
                logcp_upper_limits[m_idx] = logcp_vals[idx_limit]

    return logcp_upper_limits


def plot_multiple_marginalized_limits(
    background_events: np.ndarray,
    mu_bg: float,
    bins: int = 30,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    model: nn.Module = None,
    device: str = "cpu",
    n_realizations: int = 100,
    seed_offset: int = 0,
    posteriorbins: int = 200,
    cl: float = 0.90,
    critical_mass_line: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 6),
    linewidth: float = 2.2,
    legend_linewidth: float = 5.0,
    legend_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0),
    legend_fontsize: float = 16,
    save_path: Optional[str] = None,
):
    """
    Plot overlaid 1D marginalized upper limits from multiple realizations.
    
    Args:
        background_events: Array of (n_events, 2) with background S1, S2 values
        mu_bg: Expected number of background events in null spectrum
        bins: Number of bins in each S1/S2 dimension
        logm_range, logcp_range: Parameter grid ranges for posterior computation
        model: Trained SBI model
        device: torch device for inference
        n_realizations: Number of background fluctuation realizations to plot
        seed_offset: Random seed offset for reproducibility
        posteriorbins: Number of grid points for posterior computation
        cl: Credible level (default 0.90)
        critical_mass_line: Optional mass (GeV) to show as vertical line
        figsize: Figure dimensions
        linewidth: Line width for the plotted limits
        legend_linewidth: Line width for the UL legend handle
        legend_color: RGBA color for the UL legend handle
        legend_fontsize: Font size for the legend
        save_path: Path to save the figure (optional)
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    seeds = np.arange(n_realizations) + seed_offset

    fig, ax = plt.subplots(figsize=figsize)

    # Generate and plot upper limits for each background realization
    for i, seed in enumerate(seeds):
        # Create null spectrum with this realization of background fluctuations
        null_spec = generate_null_spectrum_s1s2(
            background_events=background_events,
            mu_bg=mu_bg,
            s1_bins=bins,
            s2_bins=bins,
            rng_seed=int(seed),
        )

        # Compute posterior for this spectrum
        posterior, logm_vals, logcp_vals = posterior_grid(
            null_spec,
            logm_range=logm_range,
            logcp_range=logcp_range,
            posteriorbins=posteriorbins,
            model=model,
            device=device,
        )

        # Extract 1D marginalized upper limit
        logcp_lim = compute_marginalized_upper_limit(posterior.detach().cpu().numpy(), logm_vals, logcp_vals, cl=cl)

        # Convert to linear space for plotting
        m_vals = 10 ** logm_vals
        cp_vals = 10 ** logcp_lim
        valid = np.isfinite(logcp_lim)

        # Plot this realization's limit curve (semi-transparent)
        ax.plot(
            m_vals[valid],
            cp_vals[valid],
            color=(0.2, 0.2, 0.2, 0.25),  # Dark gray with low alpha
            linewidth=linewidth,
            label=f"{int(cl*100)}% CL (UL)" if i == 0 else None,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.set_ylabel(r"$c_p$ [GeV$^{-2}$]")
    ax.set_xlim(1, 1000)
    ax.set_ylim(10**(-10.5), 10**(-8.5))
    ax.tick_params(axis='both', which='major', labelsize=legend_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=legend_fontsize)
    ax.xaxis.label.set_size(legend_fontsize)
    ax.yaxis.label.set_size(legend_fontsize)
    #ax.set_title(f"{int(cl*100)}% CL Upper Limits (1D Marginalized) - {n_realizations} Background Realizations")
    #ax.grid(True, which="both", ls="--", alpha=0.5)
    
    if critical_mass_line is not None:
        ax.axvline(critical_mass_line, color='red', linestyle='--', linewidth=2.0, 
                   label=f'Critical mass = {critical_mass_line:.1f} GeV', alpha=0.7)
    
    legend = ax.legend(loc="lower right", fontsize=legend_fontsize)
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    labels = [text.get_text() for text in legend.get_texts()]
    for handle, label in zip(handles, labels):
        if "(UL)" in label:
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(legend_linewidth)
            if hasattr(handle, "set_color"):
                handle.set_color(legend_color)
            if hasattr(handle, "set_alpha"):
                handle.set_alpha(legend_color[3])
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()


def compute_exclusion_statistics(
    logcp_limits_list: List[np.ndarray],
    smoothing_window: int = 10,
) -> dict:
    """Compute statistical summary of exclusion limits from multiple realizations."""
    from scipy.ndimage import uniform_filter1d

    logcp_stack = np.vstack(logcp_limits_list)
    n_bins = logcp_stack.shape[1]

    n_valid = np.sum(~np.isnan(logcp_stack), axis=0)

    logcp_mean = np.full(n_bins, np.nan)
    logcp_med = np.full(n_bins, np.nan)
    logcp_p16 = np.full(n_bins, np.nan)
    logcp_p84 = np.full(n_bins, np.nan)
    logcp_p2p5 = np.full(n_bins, np.nan)
    logcp_p97p5 = np.full(n_bins, np.nan)

    for m_idx in range(n_bins):
        col = logcp_stack[:, m_idx]
        if not np.any(np.isfinite(col)):
            continue
        logcp_mean[m_idx] = np.nanmean(col)
        logcp_med[m_idx] = np.nanpercentile(col, 50)
        logcp_p16[m_idx] = np.nanpercentile(col, 16)
        logcp_p84[m_idx] = np.nanpercentile(col, 84)
        logcp_p2p5[m_idx] = np.nanpercentile(col, 2.5)
        logcp_p97p5[m_idx] = np.nanpercentile(col, 97.5)

    if smoothing_window > 0:
        logcp_mean = uniform_filter1d(logcp_mean, size=smoothing_window, mode="nearest")
        logcp_med = uniform_filter1d(logcp_med, size=smoothing_window, mode="nearest")
        logcp_p16 = uniform_filter1d(logcp_p16, size=smoothing_window, mode="nearest")
        logcp_p84 = uniform_filter1d(logcp_p84, size=smoothing_window, mode="nearest")
        logcp_p2p5 = uniform_filter1d(logcp_p2p5, size=smoothing_window, mode="nearest")
        logcp_p97p5 = uniform_filter1d(logcp_p97p5, size=smoothing_window, mode="nearest")

    return {
        'mean': logcp_mean,
        'median': logcp_med,
        'p16': logcp_p16,
        'p84': logcp_p84,
        'p2p5': logcp_p2p5,
        'p97p5': logcp_p97p5,
        'n_valid': n_valid,
    }


def compute_marginalized_limit_statistics(
    background_events: np.ndarray,
    mu_bg: float,
    bins: int = 30,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    model: nn.Module = None,
    device: str = "cpu",
    n_realizations: int = 100,
    seed_offset: int = 0,
    posteriorbins: int = 200,
    cl: float = 0.90,
    smoothing_window: int = 10,
) -> Tuple[dict, np.ndarray, List[np.ndarray]]:
    """
    Compute statistical summary of 1D marginalized upper limits across multiple
    background realizations.
    
    Args:
        background_events: Array of (n_events, 2) with background S1, S2 values
        mu_bg: Expected number of background events per spectrum
        bins: Number of bins in S1/S2 dimensions
        logm_range, logcp_range: Parameter grid ranges for posterior computation
        model: Trained SBI model
        device: torch device for inference
        n_realizations: Number of background realizations
        seed_offset: Random seed offset for reproducibility
        posteriorbins: Grid resolution for posterior computation
        cl: Credible level for upper limits (default 0.90)
        min_coverage: Minimum fraction of valid limits required per mass bin
        smoothing_window: Optional smoothing kernel size (0 = no smoothing)
    
    Returns:
        stats: Dictionary with keys 'mean', 'median', 'p16', 'p84', 'p2p5', 'p97p5', 'n_valid'
        logm_vals: Array of log10(mass) bin centers
        logcp_limits_list: List of raw limit arrays from each realization
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    seeds = np.arange(n_realizations) + seed_offset
    logcp_limits_list = []
    logm_vals = None
    
    for seed in seeds:
        null_spec = generate_null_spectrum_s1s2(
            background_events=background_events,
            mu_bg=mu_bg,
            s1_bins=bins,
            s2_bins=bins,
            rng_seed=int(seed),
        )
        
        posterior, logm_vals_iter, logcp_vals = posterior_grid(
            null_spec,
            logm_range=logm_range,
            logcp_range=logcp_range,
            posteriorbins=posteriorbins,
            model=model,
            device=device,
        )
        
        logcp_lim = compute_marginalized_upper_limit(
            posterior.detach().cpu().numpy(),
            logm_vals_iter,
            logcp_vals,
            cl=cl,
        )
        
        if logm_vals is None:
            logm_vals = logm_vals_iter
        
        logcp_limits_list.append(logcp_lim)
    
    stats = compute_exclusion_statistics(
        logcp_limits_list,
        smoothing_window=smoothing_window,
    )
    
    return stats, logm_vals, logcp_limits_list


def determine_critical_mass(
    logm_vals: np.ndarray,
    stats: dict,
    verbose: bool = True,
) -> float:
    """
    Determine critical mass threshold below which posteriors become uninformative.
    
    Finds the mass at which the median limit curve has the steepest slope,
    i.e. the transition point from flat (uninformative) to varying (informative).
    
    Args:
        logm_vals: Array of log10(mass) bin centers [GeV]
        stats: Dictionary from compute_exclusion_statistics() with 'median' key
        verbose: Print the result
    
    Returns:
        critical_mass: Critical mass in GeV below which limits should be masked
    """
    median_limit = stats['median']
    m_vals = 10 ** logm_vals
    
    # Only work with valid (non-NaN) region
    valid_mask = ~np.isnan(median_limit)
    if np.sum(valid_mask) < 2:
        return m_vals[0]
    
    logm_valid = logm_vals[valid_mask]
    logcp_valid = median_limit[valid_mask]
    m_valid = 10 ** logm_valid
    
    # Compute absolute slope: |d(logcp)/d(logm)|
    slope = np.abs(np.gradient(logcp_valid, logm_valid))
    
    # Critical mass = mass with maximum slope (steepest transition)
    critical_mass = m_valid[np.argmax(slope)]
    
    if verbose:
        print(f"Critical Mass: {critical_mass:.1f} GeV")
    
    return critical_mass


def apply_critical_mass_threshold(
    logm_vals: np.ndarray,
    logcp_limits_list: List[np.ndarray],
    critical_mass: float,
) -> List[np.ndarray]:
    """
    Apply critical mass threshold to a list of limit curves.
    
    Sets all limits below the critical mass to NaN (masked out).
    
    Args:
        logm_vals: Array of log10(mass) bin centers
        logcp_limits_list: List of logcp limit arrays
        critical_mass: Mass threshold in GeV
    
    Returns:
        List of limit arrays with threshold applied
    """
    mass_vals = 10 ** logm_vals
    below_threshold = mass_vals < critical_mass
    
    thresholded_limits = []
    for logcp_lim in logcp_limits_list:
        logcp_lim_copy = logcp_lim.copy()
        logcp_lim_copy[below_threshold] = np.nan
        thresholded_limits.append(logcp_lim_copy)
    
    return thresholded_limits


def plot_marginalized_limits_with_bands(
    logm_vals: np.ndarray,
    stats: dict,
    show_mean: bool = True,
    show_median: bool = True,
    critical_mass: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 6),
    axis_label_fontsize: float = PLOT_AXIS_LABEL_SIZE,
    tick_label_fontsize: float = PLOT_TICK_LABEL_SIZE,
    legend_fontsize: float = PLOT_LEGEND_SIZE,
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    official_xenonnt_limit: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot 1D marginalized upper limits with uncertainty bands.
    
    Args:
        logm_vals: log10(mass) bin centers
        stats: Dictionary from compute_marginalized_limit_statistics()
        show_mean: Include mean upper limit line
        show_median: Include median upper limit line
        critical_mass: Optional mass threshold. If provided, masks out data below this value.
        figsize: Figure size
        axis_label_fontsize: Font size for x/y axis labels
        tick_label_fontsize: Font size for tick labels
        legend_fontsize: Font size for legend labels
        y_quantity: "cp" or "sigma" — y-axis quantity
        plot_official_xenonnt: If True, overlay the official XENONnT exclusion limit
        official_xenonnt_limit: Pre-loaded XENONnT limit dict (optional)
        save_path: Optional path to save the figure as PDF
    """
    y_quantity = _normalize_y_quantity(y_quantity)
    m_vals = 10 ** logm_vals

    # Filter to valid estimates (not NaN)
    valid = np.isfinite(stats['median'])

    # Apply critical mass threshold if provided
    if critical_mass is not None:
        above_threshold = m_vals >= critical_mass
        valid = valid & above_threshold

    m_vals    = m_vals[valid]
    cp_mean   = 10 ** stats['mean'][valid]
    cp_med    = 10 ** stats['median'][valid]
    cp_1lo    = 10 ** stats['p16'][valid]
    cp_1hi    = 10 ** stats['p84'][valid]
    cp_2lo    = 10 ** stats['p2p5'][valid]
    cp_2hi    = 10 ** stats['p97p5'][valid]

    if y_quantity == "sigma":
        y_mean = _cp_to_sigma_p(m_vals, cp_mean)
        y_med  = _cp_to_sigma_p(m_vals, cp_med)
        y_1lo  = _cp_to_sigma_p(m_vals, cp_1lo)
        y_1hi  = _cp_to_sigma_p(m_vals, cp_1hi)
        y_2lo  = _cp_to_sigma_p(m_vals, cp_2lo)
        y_2hi  = _cp_to_sigma_p(m_vals, cp_2hi)
    else:
        y_mean = cp_mean
        y_med  = cp_med
        y_1lo  = cp_1lo
        y_1hi  = cp_1hi
        y_2lo  = cp_2lo
        y_2hi  = cp_2hi

    fig, ax = plt.subplots(figsize=figsize)

    # Plot uncertainty bands
    ax.fill_between(m_vals, y_1lo, y_1hi, color="green",  alpha=0.50, label=r"68% band (1$\sigma$)")
    ax.fill_between(m_vals, y_2lo, y_2hi, color="yellow", alpha=0.50, label=r"95% band (2$\sigma$)")

    # Plot central estimates
    if show_mean:
        ax.plot(m_vals, y_mean, color="blue",  linewidth=2.0, linestyle="-",  label="Mean upper limit")
    if show_median:
        ax.plot(m_vals, y_med,  color="black", linewidth=3.2, linestyle="--", label="Median upper limit")

    plot_official_xenonnt_limit(
        ax,
        plot_official_xenonnt=plot_official_xenonnt,
        y_quantity=y_quantity,
        official_xenonnt_limit=official_xenonnt_limit,
        color=(0.5, 0.5, 0.5),
        linewidth=3.2,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    if y_quantity == "sigma":
        m_rep = np.sqrt(1.0 * 1000.0)
        ax.set_ylim(
            _cp_to_sigma_p(m_rep, 10 ** (-10.5)),
            _cp_to_sigma_p(m_rep, 10 ** (-8.5)),
        )
    else:
        ax.set_ylim(10 ** (-10.5), 10 ** (-8.5))
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_fontsize)
    ax.set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=tick_label_fontsize)
    #ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=legend_fontsize)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()


def plot_individual_marginalized_limits_grid(
    background_events: np.ndarray,
    mu_bg: float,
    bins: int = 30,
    logm_range: Tuple[float, float] = None,
    logcp_range: Tuple[float, float] = None,
    model: nn.Module = None,
    device: str = "cpu",
    seed_offset: int = 0,
    grid_size: int = 5,
    show_posterior: bool = True,
    show_limits: bool = True,
    posteriorbins: int = 200,
    cl: float = 0.90,
    critical_mass: Optional[float] = None,
    plot_title: Optional[str] = None,
):
    """
    Plot grid of individual 1D marginalized upper limits with posteriors.
    
    Shows a 5×5 (or custom grid_size) panel display where each panel represents
    one background realization showing:
    - Background (as contour fills): the 2D posterior p(mχ, cp | data)
    - Red curve overlaid: the 1D marginalized upper limit at each mass
    
    This visualization helps understand:
    - Which realizations have strongly constraining posteriors (tight curves)
    - Which have weak constraints (flat posteriors → missing/NaN limits)
    - How background fluctuations affect the limit variability
    
    Each panel is labeled with its random seed for reproducibility tracking.
    
    Args:
        background_events: Array of (n_events, 2) with background S1, S2 values
        mu_bg: Expected number of background events
        bins: Number of bins in S1/S2 dimensions
        logm_range, logcp_range: Posterior grid ranges
        model: Trained SBI model
        device: torch device
        seed_offset: Random seed offset
        grid_size: Grid dimension (5 = 5×5 = 25 panels)
        show_posterior: If True, show posterior as background; if False, show only limits
        show_limits: If True, overlay 1D marginalized upper-limit curves (default: True)
        posteriorbins: Grid resolution for posterior computation
        cl: Credible level for upper limits
        critical_mass: Optional mass threshold (GeV). If specified, only plot limits above this mass
        plot_title: Optional custom title (default: auto-generated from cl)
    """
    if logm_range is None:
        logm_range = DEFAULT_LOGM_RANGE
    if logcp_range is None:
        logcp_range = DEFAULT_LOGCP_RANGE
    n_realizations = grid_size * grid_size
    grid_seeds = np.arange(n_realizations) + seed_offset
    figsize = (3.0 * grid_size, 3.0 * grid_size)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, seed in enumerate(grid_seeds):
        ax = axes[idx]

        # Generate null spectrum with random background event sampling
        null_spec = generate_null_spectrum_s1s2(
            background_events=background_events,
            mu_bg=mu_bg,
            s1_bins=bins,
            s2_bins=bins,
            rng_seed=int(seed),
        )

        # Compute 2D posterior for this spectrum
        posterior, logm_vals_grid, logcp_vals_grid = posterior_grid(
            null_spec,
            logm_range=logm_range,
            logcp_range=logcp_range,
            posteriorbins=posteriorbins,
            model=model,
            device=device,
        )

        posterior_np = posterior.detach().cpu().numpy()
        m_vals_grid = 10 ** logm_vals_grid
        cp_vals_grid_full = 10 ** logcp_vals_grid

        # Show 2D posterior as background
        if show_posterior:
            ax.contourf(m_vals_grid, cp_vals_grid_full, posterior_np, levels=50, cmap="viridis", alpha=0.7)

        # Overlay the 1D marginalized upper limit
        if show_limits:
            logcp_lim_grid = compute_marginalized_upper_limit(posterior_np, logm_vals_grid, logcp_vals_grid, cl=cl)
            cp_vals_grid = 10 ** logcp_lim_grid
            valid_mask = np.isfinite(logcp_lim_grid)  # Filter out NaN limits
            
            # Apply critical mass threshold if specified
            if critical_mass is not None:
                above_threshold = m_vals_grid >= critical_mass
                valid_mask = valid_mask & above_threshold
            
            ax.plot(m_vals_grid[valid_mask], cp_vals_grid[valid_mask], color="red", linewidth=1.2, alpha=0.9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10 ** logm_range[0], 10 ** logm_range[1])
        ax.set_ylim(10 ** logcp_range[0], 10 ** logcp_range[1])
        ax.tick_params(labelsize=6)
        ax.grid(True, which="both", ls="--", alpha=0.3, linewidth=0.5)

        # Label with seed number for tracking
        ax.text(0.95, 0.05, f"{seed}", transform=ax.transAxes, fontsize=6, ha="right", va="bottom", color="gray")

    # Add axis labels to edges
    fig.text(0.5, 0.04, r"$m_\chi$ [GeV]", ha="center", fontsize=14)
    fig.text(0.04, 0.5, r"$c_p$ [GeV$^{-2}$]", va="center", rotation="vertical", fontsize=14)

    if plot_title is None:
        plot_title = f"Individual {int(cl*100)}% CL Upper Limits (1D Marginalized)"
    fig.suptitle(plot_title, fontsize=16, y=0.995)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.99])
    plt.show()



# ==============================================================================
# 6. POSTERIOR VISUALIZATION - COMPARISONS
# ==============================================================================
# Compare posteriors and exclusion plots across different dataseets

def plot_inference_comparison_1x3(
    wimpy_counts: torch.Tensor,
    s1s2_signal_counts: torch.Tensor,
    background_events: np.ndarray,
    mu_bg: float,
    bins: Optional[int] = None,
    logm_range_wimpy: Tuple[float, float] = None,
    logcp_range_wimpy: Tuple[float, float] = None,
    logm_range_s1s2: Tuple[float, float] = None,
    logcp_range_s1s2: Tuple[float, float] = None,
    model_wimpy: nn.Module = None,
    model_s1s2_only: nn.Module = None,
    model_s1s2_bg: nn.Module = None,
    device: str = "cpu",
    random_bg: bool = True,
    rng_seed: int = 42,
    posteriorbins: int = 200,
    levels: int = 50,
    cmap: str = "viridis",
    mchi_true: Optional[float] = None,
    cp_true: Optional[float] = None,
    colorbar: bool = True,
    show_exclusion_lines: bool = False,
    credible_levels: Optional[List[float]] = None,
    cl: float = 0.90,
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    save_path: Optional[str] = None,
):
    """
    Compare three posteriors side-by-side: WimPy (energy), S1S2 signal-only,
    and S1S2 signal+background.

    Args:
        wimpy_counts: Energy-space spectrum for WimPy
        s1s2_signal_counts: S1S2 signal-only spectrum (flattened)
        background_events: Background event pool (S1, S2)
        mu_bg: Expected number of background events
        bins: Number of S1/S2 bins. If None, inferred from s1s2_signal_counts length.
        logm_range_wimpy, logcp_range_wimpy: WimPy posterior grid ranges
        logm_range_s1s2, logcp_range_s1s2: S1S2 posterior grid ranges
        model_wimpy: Trained WimPy model
        model_s1s2_only: Trained S1S2 model (signal-only, mu_bg=0)
        model_s1s2_bg: Trained S1S2 model (signal+background)
        device: torch device
        random_bg: If True, draw a random background realization
        rng_seed: Fixed seed used when random_bg is False
        posteriorbins: Posterior grid resolution
        levels: Contour levels for plotting
        cmap: Matplotlib colormap
        mchi_true: True WIMP mass (GeV), if provided marks true point
        cp_true: True WIMP coupling (GeV^-2), if provided marks true point
        colorbar: If True, add one independent colorbar per panel
        show_exclusion_lines: If True, overlay exclusion lines (HPD for first two, marginalized limit for bg)
        credible_levels: List of credible levels for HPD contours (default [0.68, 0.95])
        cl: Credible level for marginalized upper limits
        y_quantity: "cp" or "sigma" — y-axis quantity
        plot_official_xenonnt: If True, overlay the official XENONnT limit on each panel
    """
    if logm_range_wimpy is None:
        logm_range_wimpy = DEFAULT_LOGM_RANGE
    if logcp_range_wimpy is None:
        logcp_range_wimpy = DEFAULT_LOGCP_RANGE
    if logm_range_s1s2 is None:
        logm_range_s1s2 = DEFAULT_LOGM_RANGE
    if logcp_range_s1s2 is None:
        logcp_range_s1s2 = DEFAULT_LOGCP_RANGE
    y_quantity = _normalize_y_quantity(y_quantity)
    if credible_levels is None:
        credible_levels = [0.68, 0.95]

    # Ensure 1D S1S2 input and infer binning when not explicitly provided.
    s1s2_signal_counts = s1s2_signal_counts.reshape(-1)
    if bins is None:
        n_s1s2 = int(s1s2_signal_counts.numel())
        inferred_bins = int(np.sqrt(n_s1s2))
        if inferred_bins * inferred_bins != n_s1s2:
            raise ValueError(
                f"Cannot infer square S1S2 binning from {n_s1s2} features. "
                "Please pass bins explicitly."
            )
        bins = inferred_bins
    
    if random_bg:
        rng_seed = int(np.random.randint(0, 1_000_000))

    bg_spec = generate_null_spectrum_s1s2(
        background_events=background_events,
        mu_bg=mu_bg,
        s1_bins=bins,
        s2_bins=bins,
        rng_seed=rng_seed,
    )
    bg_spec = torch.as_tensor(
        bg_spec,
        dtype=s1s2_signal_counts.dtype,
        device=s1s2_signal_counts.device,
    ).reshape(-1)
    if bg_spec.numel() != s1s2_signal_counts.numel():
        raise ValueError(
            f"S1S2/background shape mismatch: signal has {s1s2_signal_counts.numel()} features "
            f"but background has {bg_spec.numel()}. Use bins={int(np.sqrt(s1s2_signal_counts.numel()))}."
        )
    s1s2_bg_counts = s1s2_signal_counts + bg_spec

    post_w, logm_w, logcp_w = posterior_grid(
        wimpy_counts,
        logm_range_wimpy,
        logcp_range_wimpy,
        posteriorbins=posteriorbins,
        model=model_wimpy,
        device=device,
    )
    post_s, logm_s, logcp_s = posterior_grid(
        s1s2_signal_counts,
        logm_range_s1s2,
        logcp_range_s1s2,
        posteriorbins=posteriorbins,
        model=model_s1s2_only,
        device=device,
    )
    post_b, logm_b, logcp_b = posterior_grid(
        s1s2_bg_counts,
        logm_range_s1s2,
        logcp_range_s1s2,
        posteriorbins=posteriorbins,
        model=model_s1s2_bg,
        device=device,
    )

    axis_label_size = 26
    tick_label_size = 22
    panel_title_size = 26
    colorbar_label_size = 24
    colorbar_tick_size = 20
    legend_size = PLOT_LEGEND_SIZE + 3
    event_box_font_size = 20

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    
    # Count events for each spectrum
    n_events_w = int(wimpy_counts.sum().item())
    n_events_s = int(s1s2_signal_counts.sum().item())
    n_events_b = int(s1s2_bg_counts.sum().item())
    
    panels = [
        (post_w, logm_w, logcp_w, "WimPy", n_events_w),
        (post_s, logm_s, logcp_s, "S1S2 signal-only", n_events_s),
        (post_b, logm_b, logcp_b, "S1S2 +background", n_events_b),
    ]

    ims = []
    for idx, (ax, (post, logm_vals, logcp_vals, title, n_events)) in enumerate(zip(axes, panels)):
        m_vals = 10 ** logm_vals
        cp_vals = 10 ** logcp_vals
        post_np = post.detach().cpu().numpy()

        if y_quantity == "sigma":
            M_grid, CP_grid = np.meshgrid(m_vals, cp_vals)
            Y_grid = _cp_to_sigma_p(M_grid, CP_grid)
            im = _contourf_pdf_safe(ax, M_grid, Y_grid, post_np, levels, cmap)
            y_true_plot = _cp_to_sigma_p(mchi_true, cp_true) if (mchi_true is not None and cp_true is not None) else None
        else:
            im = _contourf_pdf_safe(ax, m_vals, cp_vals, post_np, levels, cmap)
            y_true_plot = cp_true
        
        ims.append(im)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=panel_title_size)
        ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_size)
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
        ax.grid(False)
        
        # Add text box with event count only
        textstr = f"N = {n_events} events"
        legend_alpha = plt.rcParams.get("legend.framealpha", 0.8)
        legend_face = ax.get_facecolor()
        legend_edge = ax.spines["left"].get_edgecolor()
        ax.text(
            0.05,
            0.10,
            textstr,
            transform=ax.transAxes,
            fontsize=event_box_font_size,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor=legend_face,
                edgecolor=legend_edge,
                alpha=legend_alpha,
            ),
        )
        
        # Mark true parameter point if provided (as scatter point, not star)
        if mchi_true is not None and cp_true is not None:
            ax.scatter(mchi_true, y_true_plot, color=TRUE_POINT_COLOR, s=40, label="True parameters", zorder=10)

        if plot_official_xenonnt:
            plot_official_xenonnt_limit(ax, plot_official_xenonnt=True, y_quantity=y_quantity)

        if colorbar:
            cbar = fig.colorbar(im, ax=ax)
            _colorbar_pdf_safe(cbar)
            cbar.locator = ticker.MaxNLocator(nbins=4)
            cbar.formatter = ticker.ScalarFormatter(useMathText=False)
            cbar.formatter.set_scientific(False)
            cbar.formatter.set_useOffset(False)
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=colorbar_tick_size)
            if idx == len(axes) - 1:
                cbar.set_label("Posterior density", fontsize=colorbar_label_size, labelpad=12)
        
        # Add legend if we have lines or true point
        #if mchi_true is not None and cp_true is not None:
        #    ax.legend(loc="lower right", fontsize=legend_size)

    axes[0].set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_size)

    plt.tight_layout()

    if save_path is not None:
        if not save_path.lower().endswith(".pdf"):
            save_path = f"{save_path}.pdf"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        # Keep PDF export non-transparent and rely on PDF-safe contour/colorbar
        # styling to prevent unwanted seam lines.
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=False)

    plt.show()


def plot_hpd_and_marginalized_comparison(
    wimpy_counts: torch.Tensor,
    s1s2_signal_counts: torch.Tensor,
    background_events: np.ndarray,
    mu_bg: float,
    bins: Optional[int] = None,
    logm_range_wimpy: Tuple[float, float] = None,
    logcp_range_wimpy: Tuple[float, float] = None,
    logm_range_s1s2: Tuple[float, float] = None,
    logcp_range_s1s2: Tuple[float, float] = None,
    model_wimpy: nn.Module = None,
    model_s1s2_only: nn.Module = None,
    model_s1s2_bg: nn.Module = None,
    device: str = "cpu",
    posteriorbins: int = 200,
    n_realizations: int = 50,
    seed_offset: int = 0,
    cl: float = 0.90,
    y_quantity: str = "cp",
    plot_official_xenonnt: bool = False,
    figsize: Tuple[float, float] = (8, 6),
    axis_label_fontsize: float = 12,
    tick_label_fontsize: float = 11,
    title_fontsize: float = 13,
    legend_fontsize: float = 10,
    save_path: Optional[str] = None,
):
    """
    Compare exclusion limits from three inference methods.
    
    Creates a single figure showing:
    - WimPy 90% HPD contour (single spectrum, blue line)
    - S1S2 signal-only 90% HPD contour (single spectrum, green line)
    - S1S2 signal+background limits (multiple realizations, black median + green 68% + yellow 95% bands)
      with critical mass filtering applied
    
    Args:
        wimpy_counts: Energy-space spectrum for WimPy
        s1s2_signal_counts: S1S2 signal-only spectrum (flattened)
        background_events: Background event pool (S1, S2)
        mu_bg: Expected number of background events
        bins: Number of S1/S2 bins. If None, inferred from s1s2_signal_counts length.
        logm_range_wimpy, logcp_range_wimpy: WimPy posterior grid ranges
        logm_range_s1s2, logcp_range_s1s2: S1S2 posterior grid ranges
        model_wimpy: Trained WimPy model
        model_s1s2_only: Trained S1S2 model (signal-only)
        model_s1s2_bg: Trained S1S2 model (signal+background)
        device: torch device
        posteriorbins: Posterior grid resolution
        n_realizations: Number of background realizations for uncertainty band
        seed_offset: Random seed offset for background realizations
        cl: Credible level for HPD contours and marginalized upper limits
        y_quantity: "cp" or "sigma" — y-axis quantity
        plot_official_xenonnt: Overlay official XENONnT exclusion limit
        figsize: Figure size
        axis_label_fontsize: Font size for x/y axis labels
        tick_label_fontsize: Font size for tick labels
        title_fontsize: Font size for plot title
        legend_fontsize: Font size for legend
        save_path: Optional path to save the figure as PDF
    """
    if logm_range_wimpy is None:
        logm_range_wimpy = DEFAULT_LOGM_RANGE
    if logcp_range_wimpy is None:
        logcp_range_wimpy = DEFAULT_LOGCP_RANGE
    if logm_range_s1s2 is None:
        logm_range_s1s2 = DEFAULT_LOGM_RANGE
    if logcp_range_s1s2 is None:
        logcp_range_s1s2 = DEFAULT_LOGCP_RANGE
    y_quantity = _normalize_y_quantity(y_quantity)

    # Ensure 1D S1S2 input and infer binning when not explicitly provided.
    s1s2_signal_counts = s1s2_signal_counts.reshape(-1)
    if bins is None:
        n_s1s2 = int(s1s2_signal_counts.numel())
        inferred_bins = int(np.sqrt(n_s1s2))
        if inferred_bins * inferred_bins != n_s1s2:
            raise ValueError(
                f"Cannot infer square S1S2 binning from {n_s1s2} features. "
                "Please pass bins explicitly."
            )
        bins = inferred_bins

    # Compute posteriors for WimPy and S1S2 signal-only (single spectra)
    post_w, logm_w, logcp_w = posterior_grid(
        wimpy_counts, logm_range_wimpy, logcp_range_wimpy,
        posteriorbins=posteriorbins, model=model_wimpy, device=device,
    )
    post_s, logm_s, logcp_s = posterior_grid(
        s1s2_signal_counts, logm_range_s1s2, logcp_range_s1s2,
        posteriorbins=posteriorbins, model=model_s1s2_only, device=device,
    )
    
    # Extract 90% HPD contours for WimPy
    post_w_np = post_w.detach().cpu().numpy()
    threshold_w, contours_w = compute_hpd_contours(post_w_np, logm_w, logcp_w, cl=cl)
    
    # Extract 90% HPD contours for S1S2 signal-only
    post_s_np = post_s.detach().cpu().numpy()
    threshold_s, contours_s = compute_hpd_contours(post_s_np, logm_s, logcp_s, cl=cl)
    
    # Compute statistics for S1S2 signal+background over multiple realizations
    stats, logm_bg, logcp_limits_list = compute_marginalized_limit_statistics(
        background_events=background_events,
        mu_bg=mu_bg,
        bins=bins,
        logm_range=logm_range_s1s2,
        logcp_range=logcp_range_s1s2,
        model=model_s1s2_bg,
        device=device,
        n_realizations=n_realizations,
        cl=cl,
        seed_offset=seed_offset,
        posteriorbins=posteriorbins,
    )
    
    # Determine critical mass for signal+background
    critical_mass = determine_critical_mass(logm_bg, stats, verbose=False)
    
    # Apply critical mass filtering
    m_bg = 10 ** logm_bg
    above_critical = m_bg >= critical_mass
    m_bg_f   = m_bg[above_critical]
    cp_median = 10 ** stats["median"][above_critical]
    cp_1lo    = 10 ** stats["p16"][above_critical]
    cp_1hi    = 10 ** stats["p84"][above_critical]
    cp_2lo    = 10 ** stats["p2p5"][above_critical]
    cp_2hi    = 10 ** stats["p97p5"][above_critical]

    # Transform to y_quantity
    if y_quantity == "sigma":
        y_med = _cp_to_sigma_p(m_bg_f, cp_median)
        y_1lo = _cp_to_sigma_p(m_bg_f, cp_1lo)
        y_1hi = _cp_to_sigma_p(m_bg_f, cp_1hi)
        y_2lo = _cp_to_sigma_p(m_bg_f, cp_2lo)
        y_2hi = _cp_to_sigma_p(m_bg_f, cp_2hi)
    else:
        y_med, y_1lo, y_1hi, y_2lo, y_2hi = cp_median, cp_1lo, cp_1hi, cp_2lo, cp_2hi

    # Build plot
    fig, ax = plt.subplots(figsize=figsize)

    valid = np.isfinite(y_2lo) & np.isfinite(y_2hi)
    ax.fill_between(m_bg_f[valid], y_2lo[valid], y_2hi[valid],
                    alpha=0.5, color="yellow")
    valid = np.isfinite(y_1lo) & np.isfinite(y_1hi)
    ax.fill_between(m_bg_f[valid], y_1lo[valid], y_1hi[valid],
                    alpha=0.5, color="green")
    valid = np.isfinite(y_med)
    ax.plot(m_bg_f[valid], y_med[valid], color="black", linewidth=5.5,
            linestyle="--", label="Signal+bg median")

    # WimPy HPD contour
    for contour in contours_w:
        m_c, cp_c = 10 ** contour[:, 0], 10 ** contour[:, 1]
        y_c = _cp_to_sigma_p(m_c, cp_c) if y_quantity == "sigma" else cp_c
        ax.plot(m_c, y_c, color=(0,70/255,160/255), linewidth=5.5,
                label=f"WimPy ({int(cl*100)}% HPD)")
        break

    # S1S2 signal-only HPD contour
    for contour in contours_s:
        m_c, cp_c = 10 ** contour[:, 0], 10 ** contour[:, 1]
        y_c = _cp_to_sigma_p(m_c, cp_c) if y_quantity == "sigma" else cp_c
        ax.plot(m_c, y_c, color="darkgreen", linewidth=5.5, linestyle="-", alpha=0.8,
                label=f"S1S2 signal-only ({int(cl*100)}% HPD)")
        break

    plot_official_xenonnt_limit(
        ax, plot_official_xenonnt=plot_official_xenonnt, y_quantity=y_quantity, color=(0.5, 0.5, 0.5), linewidth=5.5
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    if y_quantity == "sigma":
        m_rep = np.sqrt(1.0 * 1000.0)
        ax.set_ylim(
            _cp_to_sigma_p(m_rep, 10 ** (-10.5)),
            _cp_to_sigma_p(m_rep, 10 ** (-8.5)),
        )
    else:
        ax.set_ylim(10 ** (-10.5), 10 ** (-8.5))
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_fontsize)
    ax.set_ylabel(_get_ylabel(y_quantity), fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=tick_label_fontsize)
    #ax.set_title(f"Comparison of Exclusion Limits ({int(cl*100)}% CL)", fontsize=title_fontsize)
    #ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=legend_fontsize)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.show()



# ==============================================================================
# 7. POSTERIOR HALO COMPARISONS
# ==============================================================================
# Compare posteriors across different halo configurations

def posterior_halo_comparison(
    models,
    labels,
    mchi_true,
    cp_true,
    halo_choice,
    logm_range,
    logcp_range,
    mc_config=MCConfig(),
    top_k: int = 10,
    device: str = "cpu",
    posteriorbins: int = 200,
    levels: int = 50,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """Compare posteriors from multiple halo models side-by-side."""
    axis_label_size = 26
    tick_label_size = 22
    panel_title_size = 26
    colorbar_label_size = 24
    colorbar_tick_size = 20

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    counts, _ = generate_features(
        mchi_true, cp_true, top_k=top_k, **mc_config.__dict__,
        halo_option=halo_choice, shmpp_file="shmpp_1000.npy"
    )

    for i, (model, label) in enumerate(zip(models, labels)):
        ax = axes[i]

        x = preprocess_features(model, counts, device)
        posterior, lm, lc = posterior_grid(
            x, logm_range, logcp_range, posteriorbins, model, device
        )

        m, cp = 10**lm, 10**lc

        im = _contourf_pdf_safe(ax, m, cp, posterior.cpu(), levels=levels, cmap=cmap)

        ax.scatter(mchi_true, cp_true, color=TRUE_POINT_COLOR, s=130, zorder=10)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_size)
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)

        if i == 0:
            ax.set_ylabel(r"$c_p$ [GeV$^{-2}$]", fontsize=axis_label_size)

        if label == halo_choice:
            ax.set_title(rf"$\bf{{{label}}}$", fontsize=panel_title_size)
        else:
            ax.set_title(label, fontsize=panel_title_size)

        cbar = fig.colorbar(im, ax=ax)
        _colorbar_pdf_safe(cbar)
        if i == len(models) - 1:
            cbar.set_label("Posterior density", fontsize=colorbar_label_size, labelpad=12)
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.formatter = ticker.ScalarFormatter(useMathText=False)
        cbar.formatter.set_scientific(False)
        cbar.formatter.set_useOffset(False)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=colorbar_tick_size)

    plt.tight_layout()

    if save_path is not None:
        if not save_path.lower().endswith(".pdf"):
            save_path = f"{save_path}.pdf"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        # Keep PDF export non-transparent and rely on PDF-safe contour/colorbar
        # styling to prevent unwanted seam lines.
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=False)

    plt.show()


def posterior_halo_comparison_s1s2(
    mchi_true,
    cp_true,
    halo_choice,
    models,
    labels,
    X_test,
    T_test,
    logm_range,
    logcp_range,
    device: str = "cpu",
    posteriorbins: int = 100,
    levels: int = 50,
    credible_levels: Optional[List[float]] = None,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """Compare S1S2 posteriors from multiple models on a real histogram sample."""
    axis_label_size = 26
    tick_label_size = 22
    panel_title_size = 26
    colorbar_label_size = 24
    colorbar_tick_size = 20

    log_m_true = np.log10(mchi_true)
    log_cp_true = np.log10(cp_true)
    t_np = T_test.detach().cpu().numpy()
    dist = (t_np[:, 0] - log_m_true) ** 2 + (t_np[:, 1] - log_cp_true) ** 2
    sample_idx = int(np.argmin(dist))

    sample_feature = X_test[sample_idx].to(device)
    nevents = int(sample_feature.sum().item())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for idx, (model, label) in enumerate(zip(models, labels)):
        ax = axes[idx]

        counts = sample_feature.to(device)

        posterior, lm, lc = posterior_grid(
            counts, logm_range, logcp_range, posteriorbins, model, device
        )

        m_vals = 10 ** lm
        cp_vals = 10 ** lc
        posterior_np = posterior.cpu().numpy()

        im = _contourf_pdf_safe(ax, m_vals, cp_vals, posterior_np, levels=levels, cmap=cmap)
        ax.scatter(mchi_true, cp_true, color=TRUE_POINT_COLOR, s=130, zorder=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=axis_label_size)
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)

        if idx == 0:
            ax.set_ylabel(r"$c_p$ [GeV$^{-2}$]", fontsize=axis_label_size)

        if label == halo_choice:
            ax.set_title(rf"$\bf{{{label}}}$", fontsize=panel_title_size)
        else:
            ax.set_title(label, fontsize=panel_title_size)

        cbar = fig.colorbar(im, ax=ax)
        _colorbar_pdf_safe(cbar)
        if idx == len(models) - 1:
            cbar.set_label("Posterior density", fontsize=colorbar_label_size, labelpad=12)
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.formatter = ticker.ScalarFormatter(useMathText=False)
        cbar.formatter.set_scientific(False)
        cbar.formatter.set_useOffset(False)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=colorbar_tick_size)

        if credible_levels is not None:
            plot_hpd_contours_multi(posterior_np, lm, lc, credible_levels=credible_levels, ax=ax)

        textstr = f"N = {nevents}"
        #ax.text(
        #    0.05,
        #    0.95,
        #    textstr,
        #    transform=ax.transAxes,
        #    fontsize=12,
        #    verticalalignment="top",
        #    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        #)

    cp_log10 = np.log10(cp_true)
    plt.tight_layout()

    if save_path is not None:
        if not save_path.lower().endswith(".pdf"):
            save_path = f"{save_path}.pdf"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=False)

    plt.show()



# ==============================================================================
# 8. POSTERIOR COMPARISON TO ANALYTICAL
# ==============================================================================
# Compare posteriors and exclusion plots across different dataseets

def plot_posteriors_comparison(
    mchi_true,
    cp_true,
    posterior_analytical,
    m_vals_analytical,
    cp_vals_analytical,
    posterior_nn,
    m_vals_nn,
    cp_vals_nn,
    n_mock,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """Compare analytical and NN posteriors side-by-side."""
    axis_label_size = 22
    tick_label_size = 17
    colorbar_label_size = 17
    colorbar_tick_size = 15
    event_box_font_size = 20

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.28)

    im0 = _contourf_pdf_safe(
        ax[0],
        m_vals_analytical,
        cp_vals_analytical,
        posterior_analytical,
        levels=50,
        cmap=cmap,
    )
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_title("Analytical Posterior", fontsize=axis_label_size)
    ax[0].set_xlabel("$m_\\chi$ [GeV]", fontsize=axis_label_size)
    ax[0].set_ylabel("$c_p$ [GeV$^{-2}$]", fontsize=axis_label_size)
    ax[0].tick_params(axis="both", which="both", labelsize=tick_label_size)
    if mchi_true and cp_true:
        ax[0].scatter(mchi_true, cp_true, s=50, color=TRUE_POINT_COLOR)

    ax[0].text(
        0.05,
        0.04,
        f"N={n_mock}",
        transform=ax[0].transAxes,
        fontsize=event_box_font_size,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
    )

    cbar0 = fig.colorbar(im0, ax=ax[0])
    _colorbar_pdf_safe(cbar0)
    cbar0.locator = ticker.MaxNLocator(nbins=4)
    cbar0.formatter = ticker.FuncFormatter(lambda x, pos: f"{x:#.2g}")
    cbar0.update_ticks()
    cbar0.ax.tick_params(labelsize=colorbar_tick_size)

    im1 = _contourf_pdf_safe(
        ax[1],
        m_vals_nn,
        cp_vals_nn,
        posterior_nn.cpu().numpy(),
        levels=50,
        cmap=cmap,
    )
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_title("Neural Posterior", fontsize=axis_label_size)
    ax[1].set_xlabel("$m_\\chi$ [GeV]", fontsize=axis_label_size)
    ax[1].tick_params(axis="both", which="both", labelsize=tick_label_size)
    if mchi_true and cp_true:
        ax[1].scatter(mchi_true, cp_true, s=50, color=TRUE_POINT_COLOR)
    cbar1 = fig.colorbar(im1, ax=ax[1])
    _colorbar_pdf_safe(cbar1)
    cbar1.set_label("Posterior density", fontsize=colorbar_label_size, labelpad=10)
    cbar1.locator = ticker.MaxNLocator(nbins=4)
    cbar1.formatter = ticker.FuncFormatter(lambda x, pos: f"{x:#.2g}")
    cbar1.update_ticks()
    cbar1.ax.tick_params(labelsize=colorbar_tick_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf", transparent=False)
    plt.show()



