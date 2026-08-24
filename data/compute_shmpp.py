"""Compute and cache the SHM++ (Gaia Sausage) halo function for direct detection.

This script evaluates the halo integral η(v_min) for a specified number of v_min bins
using WimPyDD and the SHM++ velocity distribution (Evans et al. 2018), which includes
an anisotropic "sausage" component motivated by Gaia observations.

The result is saved in WimPyDD's `Halo_functions` cache directory as:

    shmpp_{n_vmin_bins}.npy

This is computationally expensive: each v_min point requires full 3D numerical
integration. Run this script once (e.g., via SLURM), then reuse the cached result
in analyses to avoid recomputation.

Usage
-----
python3 -m data.compute_shmpp 200
python3 -m data.compute_shmpp 1000 --force
"""
import argparse
import math
import os
from typing import Tuple

import numpy as np
import WimPyDD as WD
from scipy.special import erf, erfi

# ============================================================================
# SHM++ Parameters (Evans et al. 2018)
# ============================================================================


V0_SHMPP = 233.0  # Circular velocity [km/s]
VESC_SHMPP = 528.0  # Escape velocity [km/s]
ETA_SHMPP = 0.2  # Sausage fraction
BETA_SHMPP = 0.9  # Anisotropy parameter
V_SUN_ROT = np.array([11.1, 12.2, 7.3])  # Solar peculiar velocity [km/s]


# ============================================================================
# SHM++ Velocity Distribution
# ============================================================================


def shmpp_velocity_distribution(
    u: np.ndarray,
    v0: float = V0_SHMPP,
    v_esc: float = VESC_SHMPP,
    eta: float = ETA_SHMPP,
    beta: float = BETA_SHMPP,
) -> float:
    """Galactic-frame SHM++ velocity distribution with Gaia Sausage component.

    Implements the velocity distribution from Evans et al. (2018), which models
    the Milky Way dark matter halo as a superposition of an isotropic component
    and an anisotropic "sausage" component motivated by Gaia observations.

    Parameters
    ----------
    u : np.ndarray
        Velocity vector [ux, uy, uz] in Galactic frame [km/s].
    v0 : float, default=V0_SHMPP
        Circular velocity [km/s].
    v_esc : float, default=VESC_SHMPP
        Galactic escape velocity [km/s].
    eta : float, default=ETA_SHMPP
        Fraction of DM in the Sausage component (0 ≤ η ≤ 1).
    beta : float, default=BETA_SHMPP
        Sausage anisotropy parameter.

    Returns
    -------
    float
        Velocity distribution value [km/s]^-3.

    Raises
    ------
    ValueError
        If u is not a 1D array of length 3.

    References
    ----------
    Evans et al. (2018), Phys. Rev. D 99, 023012.
    """
    u = np.asarray(u)
    if u.shape != (3,):
        raise ValueError("u must be a 1D array of length 3")

    ux, uy, uz = u
    speed2 = ux**2 + uy**2 + uz**2
    speed = math.sqrt(speed2)

    # Isotropic component (Standard Halo Model)
    sigma_v = v0 / math.sqrt(2.0)
    arg = v_esc / (math.sqrt(2.0) * sigma_v)
    N_R_esc = (
        erf(arg)
        - math.sqrt(2.0 / math.pi) * (v_esc / sigma_v) * math.exp(-0.5 * (v_esc / sigma_v) ** 2)
    )
    pref_R = 1.0 / ((2.0 * math.pi * sigma_v**2) ** 1.5 * N_R_esc)
    fR = pref_R * math.exp(-0.5 * speed2 / sigma_v**2) if speed < v_esc else 0.0

    # Sausage component (anisotropic)
    denom = 2.0 * (3.0 - 2.0 * beta)
    sigma_r_sq = (3.0 * v0**2) / denom
    sigma_t_sq = (3.0 * v0**2 * (1.0 - beta)) / denom
    sigma_r = math.sqrt(sigma_r_sq)
    sigma_t = math.sqrt(sigma_t_sq)

    a = v_esc / (math.sqrt(2.0) * sigma_r)
    term1 = erf(a)
    if beta <= 0.0 or beta >= 1.0:
        term2 = 0.0
    else:
        sqrt_factor = math.sqrt((1.0 - beta) / beta)
        exp_term = math.exp(-0.5 * v_esc**2 / sigma_t_sq)
        erfi_arg = a * math.sqrt(beta / (1.0 - beta))
        term2 = sqrt_factor * exp_term * erfi(erfi_arg)
    N_S_esc = term1 - term2

    pref_S = 1.0 / (((2.0 * math.pi) ** 1.5) * sigma_r * sigma_t**2 * N_S_esc)
    vr, vtheta, vphi = ux, uy, uz
    expo = -0.5 * (
        vr**2 / sigma_r_sq + vtheta**2 / sigma_t_sq + vphi**2 / sigma_t_sq
    )
    fS = pref_S * math.exp(expo) if speed < v_esc else 0.0

    # Combined distribution
    return (1.0 - eta) * fR + eta * fS


# ============================================================================
# Main Computation Script
# ============================================================================


def main() -> None:
    """Main entry point for computing SHM++ halo function.

    Parses command-line arguments, computes the halo function via WimPyDD,
    and saves the result to disk for later use.
    """
    parser = argparse.ArgumentParser(
        description="Compute SHM++ halo function for WIMP direct detection."
    )
    parser.add_argument(
        "n_vmin_bins",
        type=int,
        help="Number of v_min bins (e.g., 200). Larger values improve accuracy but increase computation time.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculate even if output file already exists.",
    )
    args = parser.parse_args()

    # Setup paths
    n_vmin_bins = args.n_vmin_bins
    output_file = f"shmpp_{n_vmin_bins}.npy"
    output_path = os.path.join("WimPyDD", "Halo_functions", output_file)

    # Check if file already exists
    if os.path.exists(output_path) and not args.force:
        print(f"File already exists: {output_path}")
        print("Use --force to overwrite.")
        return

    print(f"\nComputing SHM++ halo function with {n_vmin_bins} v_min bins...")
    print(f"Output: {output_path}")

    # Compute halo function using WimPyDD
    vmin, delta_eta0 = WD.streamed_halo_function(
        velocity_distribution_gal=shmpp_velocity_distribution,
        v_rot_gal=np.array([0.0, V0_SHMPP, 0.0]),
        v_esc_gal=VESC_SHMPP,
        v_sun_rot=V_SUN_ROT,
        n_vmin_bin=n_vmin_bins,
        yearly_modulation=False,
        delta_eta=True,
        full_year_sampling=False,
        recalculate=True,
        outputfile=output_file,
        v0=V0_SHMPP,
        v_esc=VESC_SHMPP,
        eta=ETA_SHMPP,
        beta=BETA_SHMPP,
    )

    print(f"\nHalo function computed successfully.")
    print(f"Saved to: {output_path}\n")


if __name__ == "__main__":
    main()