"""
Script to compute and store lambda-grids for analytical Poisson posteriors in
WIMP direct detection studies:

- Builds a λ-grid over the parameter region associated with the chosen datatags.
- Uses `mc_generator` from `data.mc_generator` to compute expected
  recoil spectra for each (mχ,cp) grid point.
- Saves results to `analytical/lambda/lambda_<datatag>_bins<gridbins>.npz`
- Reuses existing grids automatically if file already exists.

Usage
-----
    python3 -m analytical.lambda_calculation --datatag low --gridbins 100
"""

import argparse
from analytical.ppg_class import PoissonPosteriorGrid
from data.mc_generator import mc_generator


def main(datatag, gridbins):
    if gridbins <= 0:
        raise ValueError("gridbins must be a positive integer")

    print(f"Computing lambda-grid for datatag='{datatag}' with {gridbins} bins...")

    PoissonPosteriorGrid.load_or_compute(
        datatag=datatag,
        gridbins=gridbins,
        mc_generator=mc_generator
    )

    print("\nLambda grid completed and saved successfully.\n")


# ----------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute λ-grid using predefined parameter levels "
                    "('low', 'mid', 'high')."
    )

    parser.add_argument(
        "--datatag", type=str, required=True, choices=["low", "mid", "high"],
        help="Choose predefined physics region: low / mid / high"
    )

    parser.add_argument(
        "--gridbins", type=int, required=True,
        help="Number of grid bins per dimension (positive integer)."
    )

    args = parser.parse_args()
    main(args.datatag, args.gridbins)