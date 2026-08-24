"""Generate simulated WIMP recoil datasets via Monte Carlo sampling.

Samples dark matter parameters (log10 mchi, log10 cp), simulates recoil spectra,
and saves results as PyTorch .pt files. Supports multiple halo models.

Halo Models
-----------
- "default": Fixed SHM parameters from WimPyDD (fastest; baseline)
- "shm":     SHM with nuisance sampling (v0, vesc, vsun)
- "shmpp":   SHM++ model (Gaia Sausage, from precomputed file)
- "lmc":     LMC-influenced halo
- "x1":      Extreme high SHM parameters (+5 sigma)
- "x2":      Extreme low SHM parameters (-5 sigma)

Usage
-----
python3 -m data.generate_dataset --n_train 300000 --datatag low --halo_option shm
python3 -m data.generate_dataset --n_train 300000 --datatag low --halo_option shmpp --shmpp_file shmpp_1000.npy
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from configs.config import MCConfig, PARAM_RANGES
from data.mc_generator import mc_generator

# ============================================================================
# Configuration Constants
# ============================================================================

TOP_K = 10
MC_CONFIG = MCConfig()

# ============================================================================
# Helper Functions
# ============================================================================


def sample_log_prior(
    n: int, logm_interval: Tuple[float, float], logcp_interval: Tuple[float, float]
) -> np.ndarray:
    """Draw uniform samples in log10-space for (mχ, cp).

    Parameters
    ----------
    n : int
        Number of samples to draw.
    logm_interval : Tuple[float, float]
        (min, max) bounds for log10(mχ) [GeV].
    logcp_interval : Tuple[float, float]
        (min, max) bounds for log10(cp) [GeV^-2].

    Returns
    -------
    np.ndarray
        Shape (n, 2) array of [log10(mχ), log10(cp)] samples.
    """
    log_mchi = np.random.uniform(logm_interval[0], logm_interval[1], n)
    log_cp = np.random.uniform(logcp_interval[0], logcp_interval[1], n)
    return np.stack([log_mchi, log_cp], axis=1)


def generate_features(
    mchi: float,
    cp: float,
    top_k: int = TOP_K,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate feature vector from WIMP recoil simulation.

    Parameters
    ----------
    mchi : float
        WIMP mass [GeV].
    cp : float
        Isoscalar coupling [GeV^-2].
    top_k : int, default=TOP_K
        Number of highest-energy events to include.
    **kwargs
        Additional arguments passed to mc_generator.

    Returns
    -------
    features : np.ndarray
        Concatenated feature vector: [histogram, nevents, top_k_energies].
    events : np.ndarray
        Individual recoil energies [keV].
    """
    events, counts, _, nevents = mc_generator(mchi, cp, **kwargs)

    events = events.astype(np.float32)
    counts = counts.astype(np.float32)

    features = [counts, np.array([nevents], dtype=np.float32)]

    if top_k > 0:
        top_events = -np.sort(-events)[:top_k]
        padded = np.pad(top_events, (0, max(0, top_k - len(top_events))), 'constant')
        features.append(padded.astype(np.float32))

    return np.concatenate(features), events


def single_simulation(
    theta: np.ndarray,
    halo_option: str,
    shmpp_file: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate single recoil spectrum for given parameters.

    Wrapper for parallel execution via multiprocessing.

    Parameters
    ----------
    theta : np.ndarray
        [log10(mχ), log10(cp)].
    halo_option : str
        Halo model name.
    shmpp_file : str or None
        SHM++ file path (if applicable).

    Returns
    -------
    features : np.ndarray
        Feature vector for this simulation.
    events : np.ndarray
        Individual event energies.
    """
    log_mchi, log_cp = theta
    mchi, cp = 10**log_mchi, 10**log_cp
    return generate_features(
        mchi,
        cp,
        top_k=TOP_K,
        halo_option=halo_option,
        shmpp_file=shmpp_file,
        **MC_CONFIG.__dict__,
    )


# ============================================================================
# Main Dataset Generation
# ============================================================================


def main() -> None:
    """Main entry point: generate, parallelize, and save datasets.

    Parses command-line arguments, samples parameter space, runs parallel
    simulations, and saves results to disk.
    """
    parser = argparse.ArgumentParser(
        description="Generate WIMP recoil datasets via Monte Carlo simulation."
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=300000,
        help="Number of samples to generate (default: 300000)",
    )
    parser.add_argument(
        "--datatag",
        type=str,
        choices=["low", "mid", "high"],
        required=True,
        help="Parameter region tag (low/mid/high coupling strength)",
    )
    parser.add_argument(
        "--halo_option",
        choices=[
            "default",
            "shm",
            "shmpp",
            "lmc",
            "x1",
            "x2",
        ],
        default="default",
        help=(
            "Halo model: 'default', 'shm', 'shmpp', 'lmc', 'x1', 'x2' "
        ),
    )
    parser.add_argument(
        "--shmpp_file",
        type=str,
        default=None,
        help="Precomputed SHM++ file (required for --halo_option=shmpp)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.halo_option == "shmpp" and args.shmpp_file is None:
        parser.error("--shmpp_file must be specified when --halo_option=shmpp")

    # Determine number of workers (SLURM-aware)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", max(1, cpu_count() - 2)))
    print(f"\nUsing {n_workers} workers on {cpu_count()} cores")

    # Prepare output directory
    save_dir = os.path.join("data", "datasets")
    os.makedirs(save_dir, exist_ok=True)

    logm_interval = PARAM_RANGES[args.datatag]["logm_range"]
    logcp_interval = PARAM_RANGES[args.datatag]["logcp_range"]
    print(f"\nGenerating dataset for cp interval {logcp_interval} ({args.datatag})")
    print(f"Halo option: {args.halo_option}")
    if args.halo_option == "shmpp":
        print(f"Using SHM++ halo file: {args.shmpp_file}")

    # Sample parameter space
    theta_samples = sample_log_prior(args.n_train, logm_interval, logcp_interval)

    # Run parallel simulations
    sim_func = partial(
        single_simulation, halo_option=args.halo_option, shmpp_file=args.shmpp_file
    )
    with Pool(n_workers) as pool:
        results = list(
            tqdm(pool.imap(sim_func, theta_samples), total=args.n_train, desc="Simulating")
        )

    # Unpack results
    features_list, events_list = zip(*results)
    features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
    theta_tensor = torch.tensor(theta_samples, dtype=torch.float32)

    # Save dataset to disk
    filename = (
        f"wimpy/{args.halo_option}/"
        f"wimpy_n{args.n_train}_{args.datatag}_{args.halo_option}.pt"
    )
    filepath = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(
        {
            "theta": theta_tensor,
            "features": features_tensor,
            "events": events_list,
            "logcp_range": logcp_interval,
            "logm_range": logm_interval,
            "n_train": args.n_train,
            "top_k": TOP_K,
            "halo_option": args.halo_option,
            "shmpp_file": args.shmpp_file,
            "mc_config": MC_CONFIG.__dict__,
        },
        filepath,
    )
    print(f"\nSaved dataset: {filepath}\n")


if __name__ == "__main__":
    main()
