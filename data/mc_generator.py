"""Monte Carlo generator for WIMP-induced nuclear recoils using WimPyDD.

Generates recoil spectra for a Xenon target under the isoscalar SI EFT Hamiltonian.
Supports different halo models via WimPyDD's `streamed_halo_function`.

Halo Options
------------
- "default": Fixed Standard Halo Model (SHM).
- "shm": Samples SHM nuisance parameters (v0, v_esc, v_sun) per call.
- "shmpp": Loads precomputed SHM++ halo file.
- "lmc": Uses precomputed Large Magellanic Cloud (LMC) halo from text file.
- "x1": Extreme high SHM parameters (+5 sigma).
- "x2": Extreme low SHM parameters (-5 sigma).

Functions
---------
- calculate_diff_rate(): Compute differential recoil rate.
- mc_generator(): Draw recoil events and histograms.
"""
import os
from typing import Optional, Tuple

import numpy as np
import WimPyDD as WD

from data.compute_shmpp import shmpp_velocity_distribution

# ============================================================================
# Physics Configuration
# ============================================================================

# Isoscalar SI Hamiltonian
WC = {"1": lambda c_p=1e-9: [2 * c_p, 0]}
SI = WD.eft_hamiltonian("SI", WC)
XENON_TARGET = WD.target("Xe")

# ============================================================================
# SHM Halo Parameters 
# ============================================================================

# Recommended conventions for reporting results from direct detection DM searches
MU_V0, SIGMA_V0 = 240.0, 8.0  # km/s
MU_VESC, SIGMA_VESC = 528.0, 24.5  # km/s
V_SUN_MEAN = np.array([11.1, 12.24, 7.25])
V_SUN_SIGMA_STAT = np.array([0.720, 0.470, 0.365])
V_SUN_SIGMA_SYS = np.array([1.0, 2.0, 0.5])
V_SUN_SIGMA = np.sqrt(V_SUN_SIGMA_STAT**2 + V_SUN_SIGMA_SYS**2)

# ============================================================================
# Halo Function Loaders
# ============================================================================

def sample_shm_parameters() -> Tuple[float, float, np.ndarray]:
    """Sample SHM halo nuisance parameters from their priors.

    Returns
    -------
    v0 : float
        Galactic rotation velocity [km/s].
    vesc : float
        Galactic escape velocity [km/s].
    v_sun : np.ndarray
        Solar peculiar velocity vector [km/s].
    """
    v0_sample = np.random.normal(MU_V0, SIGMA_V0)
    vesc_sample = np.random.normal(MU_VESC, SIGMA_VESC)
    v_sun_sample = np.random.normal(V_SUN_MEAN, V_SUN_SIGMA)
    return v0_sample, vesc_sample, v_sun_sample


def _compute_shm_halo_with_params(
    v0: float, vesc: float, v_sun: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal helper to compute SHM halo function for given parameters.

    Parameters
    ----------
    v0 : float
        Galactic rotation velocity [km/s].
    vesc : float
        Galactic escape velocity [km/s].
    v_sun : np.ndarray
        Solar peculiar velocity vector [km/s].

    Returns
    -------
    vmin : np.ndarray
        Minimum velocities for halo function.
    delta_eta0 : np.ndarray
        Speed distribution differences.
    """
    vmin, delta_eta0 = WD.streamed_halo_function(
        v_rot_gal=np.array([0.0, v0, 0.0]),
        v_esc_gal=vesc,
        v_sun_rot=v_sun,
        n_vmin_bin=1000,
        yearly_modulation=False,
        delta_eta=True,
        full_year_sampling=False,
        recalculate=False,
    )
    return vmin, delta_eta0


def compute_shm_halo() -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHM halo function with randomly sampled nuisance parameters.

    Returns
    -------
    vmin : np.ndarray
        Minimum velocities for halo function.
    delta_eta0 : np.ndarray
        Speed distribution differences.
    """
    v0, vesc, v_sun = sample_shm_parameters()
    return _compute_shm_halo_with_params(v0, vesc, v_sun)


def compute_shm_halo_sigma(sigma_offset: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHM halo with a fixed sigma offset.

    Returns
    ------- 
    vmin : np.ndarray
        Minimum velocities for halo function.
    delta_eta0 : np.ndarray
        Speed distribution differences.
    """
    v0 = MU_V0 + sigma_offset * SIGMA_V0
    vesc = MU_VESC + sigma_offset * SIGMA_VESC
    v_sun = V_SUN_MEAN
    return _compute_shm_halo_with_params(v0, vesc, v_sun)


def compute_shm_halo_x1() -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHM halo with extreme high (+5sigma) velocity parameters.

    Returns
    -------
    vmin : np.ndarray
        Minimum velocities for halo function.
    delta_eta0 : np.ndarray
        Speed distribution differences.
    """
    return compute_shm_halo_sigma(5)


def compute_shm_halo_x2() -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHM halo with extreme low (-5sigma) velocity parameters.

    Returns
    -------
    vmin : np.ndarray
        Minimum velocities for halo function.
    delta_eta0 : np.ndarray
        Speed distribution differences.
    """
    return compute_shm_halo_sigma(-5)


def load_shmpp_halo(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed SHM++ halo function.

    Parameters
    ----------
    filename : str
        Name of the SHM++ halo file in WimPyDD/Halo_functions/.

    Returns
    -------
    vmin : np.ndarray
        Minimum velocities for halo function.
    delta_eta0 : np.ndarray
        Speed distribution differences.

    Raises
    ------
    FileNotFoundError
        If the halo file does not exist.
    """
    halo_path = os.path.join("WimPyDD", "Halo_functions", filename)
    if not os.path.exists(halo_path):
        raise FileNotFoundError(
            f"Halo file {halo_path} not found. Did you generate it?"
        )

    # Standard SHM++ parameters (Evans et al. 2018)
    v0 = 233.0
    vesc = 528.0
    eta = 0.2
    beta = 0.9
    v_sun_rot = np.array([11.1, 12.2, 7.3])

    vmin, delta_eta0 = WD.streamed_halo_function(
        velocity_distribution_gal=shmpp_velocity_distribution,
        v_rot_gal=np.array([0.0, v0, 0.0]),
        v_esc_gal=vesc,
        v_sun_rot=v_sun_rot,
        n_vmin_bin=1000,
        yearly_modulation=False,
        delta_eta=True,
        full_year_sampling=False,
        recalculate=False,
        outputfile=filename,
        v0=v0,
        v_esc=vesc,
        eta=eta,
        beta=beta,
    )

    return vmin, delta_eta0


def load_lmc_halo() -> Tuple[np.ndarray, np.ndarray]:
    """Load Large Magellanic Cloud enhanced halo model.

    Uses Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) for
    monotonic interpolation of the LMC velocity distribution.

    Returns
    -------
    vmin : np.ndarray
        Minimum velocities for halo function.
    delta_eta0 : np.ndarray
        Speed distribution differences.
    """
    from scipy.interpolate import PchipInterpolator 

    # Load raw LMC data
    data = np.loadtxt("WimPyDD/Halo_functions/lmc.txt")
    # Columns: vmin, eta_median, eta_lower, eta_upper, h
    vmin_raw = data[:, 0]
    eta_median_raw = data[:, 1]

    interp_eta = PchipInterpolator(vmin_raw, eta_median_raw)
    vmin_fine = np.linspace(vmin_raw.min(), vmin_raw.max(), 1000)
    eta_median_fine = interp_eta(vmin_fine)
    eta_median_fine = np.clip(eta_median_fine, 0, None)

    # Compute delta_eta0
    delta_eta0 = np.zeros_like(eta_median_fine)
    delta_eta0[1:] = eta_median_fine[:-1] - eta_median_fine[1:]

    return vmin_fine, delta_eta0


# ============================================================================
# Differential Rate Calculation
# ============================================================================


def calculate_diff_rate(
    mchi: float,
    c_p: float,
    er_vec: np.ndarray,
    halo_option: str = "default",
    shmpp_file: Optional[str] = None,
) -> np.ndarray:
    """Calculate differential nuclear recoil rate dR/dE.

    Computes the differential recoil spectrum [events/(kg·day·keV)] for a Xenon
    target using the specified dark matter halo model.

    Parameters
    ----------
    mchi : float
        WIMP mass [GeV].
    c_p : float
        Isoscalar coupling strength [GeV^-2].
    er_vec : np.ndarray
        Recoil energies [keV] at which to evaluate the rate.
    halo_option : str, default="default"
        Halo model choice:
        - "default": WimPyDD default streamed halo.
        - "shm": Standard Halo Model with sampled nuisance parameters.
        - "shmpp": Precomputed SHM++ halo (requires shmpp_file).
        - "lmc": Large Magellanic Cloud enhanced halo.
        - "x1": Extreme high SHM parameters (+5 sigma).
        - "x2": Extreme low SHM parameters (-5 sigma).
    shmpp_file : str, optional
        SHM++ halo filename (required if halo_option="shmpp").

    Returns
    -------
    np.ndarray
        Differential recoil rate for each energy, [events/(kg·day·keV)].

    Raises
    ------
    ValueError
        If halo_option is unknown or shmpp_file not provided when required.
    """
    # Load halo function based on option
    if halo_option == "default":
        vmin, delta_eta0 = WD.streamed_halo_function()
    elif halo_option == "shm":
        vmin, delta_eta0 = compute_shm_halo()
    elif halo_option == "shmpp":
        if shmpp_file is None:
            raise ValueError(
                "For halo_option='shmpp', shmpp_file must be provided"
            )
        vmin, delta_eta0 = load_shmpp_halo(shmpp_file)
    elif halo_option == "lmc":
        vmin, delta_eta0 = load_lmc_halo()
    elif halo_option == "x1":
        vmin, delta_eta0 = compute_shm_halo_x1()
    elif halo_option == "x2":
        vmin, delta_eta0 = compute_shm_halo_x2()
    else:
        raise ValueError(f"Unknown halo_option: {halo_option}")

    # Compute differential rate for each energy
    return np.array(
        [
            WD.diff_rate(XENON_TARGET, SI, mchi, er, vmin, delta_eta0, c_p=c_p)
            for er in er_vec
        ]
    )


# ============================================================================
# Monte Carlo Event Generator
# ============================================================================


def mc_generator(
    mchi: float,
    cp: float,
    exposure: float = 365000,
    low: float = 1.0,
    up: float = 100.0,
    bins: int = 100,
    log_bins: bool = True,
    poisson: bool = True,
    halo_option: str = "default",
    shmpp_file: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, float]:
    """Generate WIMP recoil events and histogram.

    Parameters
    ----------
    mchi : float
        WIMP mass [GeV].
    cp : float
        Isoscalar coupling strength [GeV^-2].
    exposure : float, default=365000
        Detector exposure [kg·days].
    low : float, default=1.0
        Lower bound of recoil energy range [keV].
    up : float, default=100.0
        Upper bound of recoil energy range [keV].
    bins : int, default=100
        Number of histogram bins.
    log_bins : bool, default=True
        Use logarithmic binning if True, linear otherwise.
    poisson : bool, default=True
        Apply Poisson fluctuations if True; return expectation if False.
    halo_option : str, default="default"
        Halo model (see `calculate_diff_rate` for options).
    shmpp_file : str, optional
        SHM++ halo filename (required if halo_option="shmpp").

    Returns
    -------
    events : np.ndarray or None
        Generated recoil energies [keV]. None if poisson=False.
    counts : np.ndarray
        Histogram bin counts (or expected counts if poisson=False).
    bin_edges : np.ndarray
        Histogram bin edges [keV].
    nevents : float
        Total number of events (sampled or expected).
    """

    # Create bin edges
    if log_bins:
        er_vec_edges = np.logspace(np.log10(low), np.log10(up), bins + 1)
    else:
        er_vec_edges = np.linspace(low, up, bins + 1)

    bin_widths = np.diff(er_vec_edges)
    er_vec = 0.5 * (er_vec_edges[:-1] + er_vec_edges[1:])

    # Compute expected events per bin
    diff_rate = calculate_diff_rate(mchi, cp, er_vec, halo_option, shmpp_file)
    expected_events_per_bin = diff_rate * bin_widths * exposure
    n_expected = expected_events_per_bin.sum()

    # Return expectation values if not using Poisson sampling
    if not poisson:
        return None, expected_events_per_bin, er_vec_edges, n_expected

    # Sample total number of events
    nevents = np.random.poisson(n_expected)

    # Handle zero-event case
    if n_expected == 0 or nevents == 0:
        return np.array([]), np.zeros(bins, dtype=int), er_vec_edges, 0.0

    # Build normalized cumulative distribution function (CDF)
    cdf = np.cumsum(expected_events_per_bin)
    cdf /= cdf[-1]

    # Sample events by inverting the CDF
    u = np.random.uniform(0, 1, nevents)
    events = np.interp(u, cdf, er_vec)

    # Histogram the sampled events
    counts, bin_edges = np.histogram(events, bins=er_vec_edges)

    return events, counts, bin_edges, nevents



