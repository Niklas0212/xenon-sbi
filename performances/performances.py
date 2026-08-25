"""
Performance evaluation for trained SBI models on dark matter detection tasks.

Provides multi-metric evaluation (coverage, JSD, Euclidean/Mahalanobis),
result caching, and formatted output.

The command-line interface evaluates the selected WimPyDD models for one or
more energy regimes. Results are stored as PyTorch files in
``performances/performances_results`` and reused when the same evaluation
configuration is requested again.

Usage:
    python3 -m performances.performances --n-total 20000
    python3 -m performances.performances --model ntothighest --datatags low mid
"""

import os
import sys
import argparse
import torch
import numpy as np
from tabulate import tabulate

from data.mc_generator import mc_generator
from analytical.ppg_class import PoissonPosteriorGrid
from utils.processing import load_matching_pairs

from performances.metrics import (
    coverage_test,
    jsd_eval,
    euclidean_mahalanobis_eval,
)
from configs.config import load_model, MODEL_CONFIG, PARAM_RANGES

# ============================================================
# Core evaluation
# ============================================================

def evaluate_models(
    datatags,
    n_train=300_000,
    device="cpu",
    n_total=500,
    split="val",
    halo="default",
    specific_model=None,
):
    """
    Evaluate all or specific models on multiple performance metrics.

    Loads matching data pairs, analytical posterior grids, and trained models,
    then computes coverage, JSD, and distance-based metrics.

    Parameters
    ----------
    datatags : list[str]
        Energy regimes to evaluate (e.g., ["low", "mid", "high"])
    n_train : int, optional
        Number of training samples used for model training (default: 300_000)
    device : str, optional
        Device for inference ("cpu" or "cuda", default: "cpu")
    n_total : int, optional
        Number of evaluation samples per model (default: 500)
    split : str, optional
        Dataset split ("val" or "test", default: "val")
    halo : str, optional
        Halo configuration name (default: "default")
    specific_model : str or list[str], optional
        Specific model(s) to evaluate. If None, evaluates all models (default: None)

    Returns
    -------
    dict[str, list[dict]]
        Results organized by datatag. Each entry contains list of dicts with keys:
        "model", "val_loss", "val_acc", "cvg_abs", "cvg_signed", "jsd",
        "euclidean_median", "euclidean_mean", "mahalanobis_median", "mahalanobis_mean"
        Models whose checkpoint does not exist are omitted from the corresponding
        datatag.
    """

    results_by_tag = {tag: [] for tag in datatags}

    for datatag in datatags:

        # -----------------------
        # Load dataset
        # -----------------------
        X_matching, T_matching = load_matching_pairs(
            halo_choice=halo, n_train=n_train, datatag=datatag, split=split
        )

        # -----------------------
        # Poisson posterior grid
        # -----------------------
        ppg = PoissonPosteriorGrid.load_or_compute(
            datatag, 100, mc_generator
        )

        # -----------------------
        # Decide which models
        # -----------------------
        if specific_model is not None:
            modelnames = specific_model if isinstance(specific_model, list) else [specific_model]
        else:
            modelnames = MODEL_CONFIG.keys()


        # -----------------------
        # Evaluate
        # -----------------------
        logm_range = PARAM_RANGES[datatag]["logm_range"]
        logcp_range = PARAM_RANGES[datatag]["logcp_range"]

        for modelname in modelnames:
        
            MODELPATH = f"models/wimpy/{halo}/{modelname}_n{n_train}_{datatag}_{halo}.pt"

            if not os.path.exists(MODELPATH):
                continue

            print(f"[INFO] Evaluating {MODELPATH}")

            model, ckpt = load_model(
                MODELPATH, modelname, print_arch=False
            )
            model.to(device).eval()

            # -------- Metrics --------
            _, cvg_abs, cvg_signed = coverage_test(
                model,
                X_matching,
                T_matching,
                logm_range,
                logcp_range,
                posteriorbins=100,
                device=device,
                n_samples=n_total,
                levels=np.linspace(0, 1, 50),
            )

            mean_jsd, _ = jsd_eval(
                model,
                X_matching,
                T_matching,
                logm_range,
                logcp_range,
                ppg,
                n_samples=n_total,
            )

            dist_results = euclidean_mahalanobis_eval(
                model,
                X_matching,
                T_matching,
                logm_range,
                logcp_range,
                posteriorbins=100,
                device=device,
                n_samples=n_total,
            )

            results_by_tag[datatag].append(
                dict(
                    model=modelname,
                    val_loss=ckpt["best_val_loss"],
                    val_acc=ckpt["best_val_acc"],
                    cvg_abs=cvg_abs,
                    cvg_signed=cvg_signed,
                    jsd=mean_jsd,
                    euclidean_median=dist_results["median_euclidean"],
                    euclidean_mean=dist_results["mean_euclidean"],
                    mahalanobis_median=dist_results["median_mahalanobis"],
                    mahalanobis_mean=dist_results["mean_mahalanobis"],
                )
            )

    return results_by_tag



# ============================================================
# Cache wrapper
# ============================================================

def get_or_evaluate_performances(
    datatags,
    n_train=300_000,
    device="cpu",
    n_total=500,
    split="val",
    halo="default",
    specific_model=None,
):
    """
    Get evaluation results, loading from cache if available or computing fresh.

    Wraps evaluate_models() with result caching to avoid redundant computations.
    Results are cached to disk with metadata for reproducibility.
    The cache filename includes the halo, model selection, training-set size,
    evaluation-set size, and split. Existing cache files are loaded without
    recomputing the metrics.

    Parameters
    ----------
    datatags : list[str]
        Energy regimes to evaluate
    n_train : int, optional
        Number of training samples (default: 300_000)
    device : str, optional
        Device for inference (default: "cpu")
    n_total : int, optional
        Number of evaluation samples (default: 500)
    split : str, optional
        Dataset split to evaluate (default: "val")
    halo : str, optional
        Halo configuration (default: "default")
    specific_model : str or list[str], optional
        Specific model(s) to evaluate (default: None)

    Returns
    -------
    tuple[dict, dict]
        (results, meta) where results is organized by datatag and meta contains
        evaluation parameters for reference. Results are saved under
        ``performances/performances_results/perf_<halo>_<model>_n<n_train>``
        followed by the evaluation size and split.
    """
    outdir = "performances/performances_results"
    os.makedirs(outdir, exist_ok=True)

    tag = (
        f"{specific_model}"
        if specific_model
        else "all"
    )
    filename = f"perf_{halo}_{tag}_n{n_train}_N{n_total}_{split}.pt"
    filepath = os.path.join(outdir, filename)

    if os.path.exists(filepath):
        print(f"[INFO] Loading cached results: {filepath}")
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        return data["results"], data["meta"]

    results = evaluate_models(
        datatags=datatags,
        n_train=n_train,
        device=device,
        n_total=n_total,
        split=split,
        halo=halo,
        specific_model=specific_model,
    )

    meta = dict(
        n_train=n_train,
        n_total=n_total,
        split=split,
        halo=halo,
        specific_model=specific_model,
    )

    torch.save({"results": results, "meta": meta}, filepath)
    print(f"[INFO] Saved results → {filepath}")

    return results, meta



# ============================================================
# Pretty printing
# ============================================================

def print_performance_tables(results, meta, datatags):
    """
    Print formatted performance tables for results across all datatags.

    Displays evaluation metadata and a formatted table for each datatag,
    sorted by validation accuracy (descending) and validation loss (ascending).

    Parameters
    ----------
    results : dict[str, list[dict]]
        Evaluation results organized by datatag
    meta : dict
        Metadata about the evaluation (n_train, n_total, split, etc.)
    datatags : list[str]
        Datatags to display. Each tag must be present in ``results``.

    Returns
    -------
    None
        Prints one sorted table per datatag and does not modify ``results``.
    """
    print("\nMeta:", meta)

    for tag in datatags:
        print(f"\n=== {tag.upper()} ===")

        rows = sorted(
            results[tag],
            key=lambda r: (-r["val_acc"], r["val_loss"]),
        )

        table = [
            [
                r["model"],
                r["val_loss"],
                r["val_acc"],
                r["cvg_abs"],
                r["cvg_signed"],
                r["jsd"],
                r["euclidean_median"],
                r["euclidean_mean"],
                r["mahalanobis_median"],
                r["mahalanobis_mean"],
            ]
            for r in rows
        ]

        print(tabulate(
            table,
            headers=[
                "Model", "ValLoss", "ValAcc",
                "CvgAbs", "CvgSigned",
                "JSD", "EucMed", "EucMean",
                "MahMed", "MahMean",
            ],
            floatfmt=".3f",
            tablefmt="fancy_grid",
        ))



# ============================================================
# Command Line Interface
# ============================================================

def parse_args():
    """
    Parse command line arguments for the evaluation script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with defaults. ``datatags`` contains one or more
        energy-regime names, and ``model`` is ``None`` unless a specific model
        was requested.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained SBI models on performance metrics."
    )

    parser.add_argument(
        "--n-train",
        type=int,
        default=300_000,
        help="Number of training samples used for the model (default: 300000)",
    )

    parser.add_argument(
        "--n-total",
        type=int,
        default=5_000,
        help="Number of evaluation samples (default: 5000)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on (val or test)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (cpu or cuda)",
    )

    parser.add_argument(
        "--halo",
        type=str,
        default="default",
        help="Halo configuration name",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name to evaluate (e.g. ntothighest)",
    )

    parser.add_argument(
        "--datatags",
        type=str,
        nargs="+",
        default=["low", "mid", "high"],
        help="Data tags to evaluate (default: low mid high)",
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the performance evaluation script.

    Parses command-line arguments, validates inputs, performs model evaluation,
    and prints formatted results.
    """
    args = parse_args()

    # Validate arguments
    if args.n_train <= 0 or args.n_total <= 0:
        raise ValueError("n_train and n_total must be positive integers")

    # Print evaluation configuration
    print("\n" + "=" * 60)
    print("SBI Model Performance Evaluation")
    print("=" * 60)
    print(f"Training samples:   {args.n_train:,}")
    print(f"Evaluation samples: {args.n_total:,}")
    print(f"Split:              {args.split}")
    print(f"Device:             {args.device}")
    print(f"Halo:               {args.halo}")
    print(f"Data tags:          {', '.join(args.datatags)}")
    if args.model:
        print(f"Specific model:     {args.model}")
    print("=" * 60 + "\n")

    # Evaluate and print results
    results, meta = get_or_evaluate_performances(
        datatags=args.datatags,
        n_train=args.n_train,
        device=args.device,
        n_total=args.n_total,
        split=args.split,
        halo=args.halo,
        specific_model=args.model,
    )

    print_performance_tables(results, meta, args.datatags)


if __name__ == "__main__":
    main()
