"""Project-wide configuration for WIMP direct-detection simulations and model training.

This module centralizes the default settings used throughout the analysis
pipeline, including Monte Carlo simulation parameters, parameter-space ranges,
and model-specific training configuration. It defines the reusable dataclasses
and dictionaries required to keep the experimental setup, parameter grid, and
network architectures consistent across scripts and notebooks.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from utils.architectures import (
    Full_MLP,
    Highest_MLP,
    Hist_MLP,
    HistS1S2_MLP,
    Ntot_Highest_MLP,
    Ntot_Highest_MLP_Vanilla,
    Ntot_MLP,
    S1S2_signal,
    S1S2_signal_bg,
)

# ============================================================================
# Monte Carlo Configuration
# ============================================================================

@dataclass
class MCConfig:
    """Monte Carlo simulation parameters.

    Attributes
    ----------
    exposure : int
        Detector exposure in kg·days (default: 365000).
    low : float
        Lower bound for recoil energy binning in keV.
    up : float
        Upper bound for recoil energy binning in keV.
    bins : int
        Number of histogram bins.
    log_bins : bool
        Whether to use logarithmic binning.
    poisson : bool
        Whether to apply Poisson fluctuations to event counts.
    """
    exposure: int = 365000
    low: float = 1.0
    up: float = 100.0
    bins: int = 100
    log_bins: bool = True
    poisson: bool = True


@dataclass
class TrainingDefaults:
    """Shared default hyperparameters used by training scripts."""
    top_k: int = 10
    batch_size_train: int = 4096
    batch_size_eval: int = 4096
    s1s2_bins: int = 10
    epochs: int = 150
    patience: int = 20


# ============================================================================
# Parameter Ranges
# ============================================================================

PARAM_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "low": {
        "logm_range": (0.0, 3.0),
        "logcp_range": (-10.5, -8.5),
    },
    "mid": {
        "logm_range": (0.0, 3.0),
        "logcp_range": (-10.0, -8.0),
    },
    "high": {
        "logm_range": (0.0, 3.5),
        "logcp_range": (-9.5, -7.5),
    },
}


# ============================================================================
# Model Configurations
# ============================================================================

MODEL_CONFIG = {
    "full": {
        "cls": Full_MLP,
        "input_dim_fn": lambda bins, top_k: bins + 1 + top_k,
        "optimizer": {"type": "adamw", "lr": 5e-4, "weight_decay": 1e-3, "lr_decay": 0.95},
        "hidden_dim": 64,
        "num_layers": 7,
        "dropout": 0.0,
        "batchnorm": True,
    },
    "ntothighest": {
        "cls": Ntot_Highest_MLP,
        "input_dim_fn": lambda bins, top_k: 1 + top_k,
        "optimizer": {"type": "adamw", "lr": 3e-3, "weight_decay": 1e-6, "lr_decay": 0.95},
        "hidden_dim": 512,
        "num_layers": 8,
        "dropout": 0.05,
        "batchnorm": True,
    },
    "vanilla": {
        "cls": Ntot_Highest_MLP_Vanilla,
        "input_dim_fn": lambda bins, top_k: 1 + top_k,
        "optimizer": {"type": "adam", "lr": 1e-3, "lr_decay": 0.97},
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.00,
        "batchnorm": False,
    },
    "hist": {
        "cls": Hist_MLP,
        "input_dim_fn": lambda bins, top_k: bins,
        "optimizer": {"type": "adam", "lr": 1e-3, "lr_decay": 0.97},
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.05,
        "batchnorm": True,
    },
    "ntot": {
        "cls": Ntot_MLP,
        "input_dim_fn": lambda bins, top_k: 1,
        "optimizer": {"type": "adam", "lr": 1e-3, "lr_decay": 0.97},
        "hidden_dim": 64,
        "num_layers": 4,
        "dropout": 0.05,
        "batchnorm": True,
    },
    "highest": {
        "cls": Highest_MLP,
        "input_dim_fn": lambda bins, top_k: top_k,
        "optimizer": {"type": "adam", "lr": 1e-3, "lr_decay": 0.97},
        "hidden_dim": 64,
        "num_layers": 4,
        "dropout": 0.05,
        "batchnorm": True,
    },
    "hist_s1s2": {
        "cls": HistS1S2_MLP,
        "input_dim_fn": lambda bins_s1, bins_s2: bins_s1 * bins_s2,
        "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 3e-5, "lr_decay": 0.97},
        "hidden_dim": 128,
        "num_layers": 6,
        "dropout": 0.05,
        "batchnorm": True,
    },
    "S1S2_signal": {
        "cls": S1S2_signal,
        "input_dim_fn": lambda bins_s1, bins_s2: bins_s1 * bins_s2,
        "optimizer": {"type": "adamw", "lr": 3e-3, "weight_decay": 1e-3, "lr_decay": 0.95},
        "hidden_dim": 128,
        "num_layers": 5,
        "dropout": 0.1,
        "batchnorm": True,
    },
    "S1S2_signal_bg": {
        "cls": S1S2_signal_bg,
        "input_dim_fn": lambda bins_s1, bins_s2: bins_s1 * bins_s2,
        "optimizer": {"type": "adamw", "lr": 3e-3, "weight_decay": 0.0, "lr_decay": 0.95},
        "hidden_dim": 64,
        "num_layers": 6,
        "dropout": 0.2,
        "batchnorm": True,
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Create optimizer from model config.

    Parameters
    ----------
    model : torch.nn.Module
        Model to optimize.
    cfg : dict
        Model configuration dictionary containing 'optimizer' key.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer instance.

    Raises
    ------
    ValueError
        If optimizer type is not recognized.
    """
    opt_cfg = cfg["optimizer"]
    opt_type = opt_cfg["type"]

    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(),  lr=opt_cfg["lr"])
    elif opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0))
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def get_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create optional LR scheduler from model config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer instance to schedule.
    cfg : dict
        Model configuration dictionary containing optional optimizer.lr_decay.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler | None
        ExponentialLR scheduler if lr_decay is configured, otherwise None.
    """
    gamma = cfg.get("optimizer", {}).get("lr_decay")
    if gamma is None:
        return None

    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def _load_model_from_checkpoint(
    path: str,
    modelname: str,
    input_dim: int,
    device: str,
    print_arch: bool,
) -> Tuple[torch.nn.Module, dict]:
    """Internal helper to load model from checkpoint.

    Parameters
    ----------
    path : str
        Path to checkpoint file.
    modelname : str
        Model name key in MODEL_CONFIG.
    input_dim : int
        Input dimension for the model.
    device : str
        Device to load model onto.
    print_arch : bool
        Whether to print model architecture.

    Returns
    -------
    model : torch.nn.Module
        Loaded model in eval mode.
    ckpt : dict
        Full checkpoint dictionary.
    """
    cfg = MODEL_CONFIG[modelname]
    model_args = {
        k: v for k, v in cfg.items() if k not in {"cls", "optimizer", "input_dim_fn"}
    }

    model = cfg["cls"](input_dim=input_dim, **model_args).to(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if print_arch:
        print(model)

    return model, ckpt


def load_model(
    path: str,
    modelname: str,
    bins: int = 100,
    top_k: int = 10,
    device: str = "cpu",
    print_arch: bool = False,
) -> Tuple[torch.nn.Module, dict]:
    """Load a WIMP detection model from checkpoint.

    Parameters
    ----------
    path : str
        Path to model checkpoint (.pt file).
    modelname : str
        Model architecture name (must exist in MODEL_CONFIG).
    bins : int
        Number of histogram bins (default: 100).
    top_k : int
        Number of top events to include (default: 10).
    device : str
        Device to load model onto (default: 'cpu').
    print_arch : bool
        Whether to print model architecture (default: False).

    Returns
    -------
    model : torch.nn.Module
        Loaded model in eval mode.
    ckpt : dict
        Full checkpoint dictionary.
    """
    cfg = MODEL_CONFIG[modelname]
    input_dim = cfg["input_dim_fn"](bins, top_k)
    return _load_model_from_checkpoint(path, modelname, input_dim, device, print_arch)


def load_halo_models(
    halos: List[str],
    modelname: str,
    n_train: int,
    datatag: str,
    bins: int,
    top_k: int,
    device: str = "cpu",
    base_model_dir: str = "models/wimpy",
    print_arch: bool = False,
) -> Tuple[List[torch.nn.Module], List[str]]:
    """Load individual halo models and their combined model.

    Parameters
    ----------
    halos : List[str]
        List of halo model names (e.g., ['shm', 'x1', 'x2']).
    modelname : str
        Model architecture name.
    n_train : int
        Number of training samples.
    datatag : str
        Parameter region tag ('low', 'mid', 'high').
    bins : int
        Number of histogram bins.
    top_k : int
        Number of top events.
    device : str
        Device to load models onto (default: 'cpu').
    base_model_dir : str
        Root directory for model checkpoints (default: 'models/wimpy').
    print_arch : bool
        Whether to print model architectures (default: False).

    Returns
    -------
    model_list : List[torch.nn.Module]
        [halo_1, halo_2, ..., combined] (combined only if len(halos) > 1).
    label_list : List[str]
        Corresponding labels in the same order.

    Raises
    ------
    ValueError
        If no halos are specified.
    FileNotFoundError
        If combined model path does not exist when multiple halos are provided.
    """

    if len(halos) < 1:
        raise ValueError("At least one halo must be specified.")

    model_list = []
    label_list = []

    # Load individual halo models
    for halo in halos:
        model_path = os.path.join(
            base_model_dir, halo, f"{modelname}_n{n_train}_{datatag}_{halo}.pt"
        )
        print(f"Loading model from {model_path}")

        model, _ = load_model(
            model_path,
            modelname,
            bins,
            top_k,
            device,
            print_arch=print_arch,
        )

        model_list.append(model)
        label_list.append(halo)

    # Load combined model if multiple halos
    if len(halos) > 1:
        combined_tag = "_".join(halos)
        combined_path = os.path.join(
            base_model_dir,
            "combined",
            combined_tag,
            f"{modelname}_n{n_train}_{datatag}_{combined_tag}.pt",
        )
        print(f"Loading combined model from {combined_path}")

        if not os.path.exists(combined_path):
            raise FileNotFoundError(f"Combined model not found at {combined_path}")

        model_combined, _ = load_model(
            combined_path,
            modelname,
            bins,
            top_k,
            device,
            print_arch=print_arch,
        )

        model_list.append(model_combined)
        label_list.append("combined")

    return model_list, label_list


def load_model_s1s2(
    path: str,
    bins: int,
    modelname: str,
    device: str = "cpu",
    print_arch: bool = False,
) -> Tuple[torch.nn.Module, dict]:
    """Load an S1-S2 2D histogram model from checkpoint.

    Parameters
    ----------
    path : str
        Path to model checkpoint (.pt file).
    bins : int
        Number of bins per dimension (S1 and S2).
    modelname : str
        Model architecture name (must exist in MODEL_CONFIG).
    device : str
        Device to load model onto (default: 'cpu').
    print_arch : bool
        Whether to print model architecture (default: False).

    Returns
    -------
    model : torch.nn.Module
        Loaded model in eval mode.
    ckpt : dict
        Full checkpoint dictionary.
    """
    cfg = MODEL_CONFIG[modelname]
    input_dim = cfg["input_dim_fn"](bins, bins)
    return _load_model_from_checkpoint(path, modelname, input_dim, device, print_arch)


def load_halo_models_s1s2(
    halos: List[str],
    modelname: str,
    n_train: int,
    bins: int,
    mu_bg: float = 0,
    training_mode: str = "offline",
    device: str = "cpu",
    print_arch: bool = False,
) -> Tuple[List[torch.nn.Module], List[str]]:
    """Load individual S1S2 halo models and their combined model.

    Uses the same directory convention as the S1S2 training scripts:
      Single halo:  models/xenon/{mode}/{signal_type}/{halo}/{model}.pt
      Combined:     models/xenon/{mode}/{signal_type}/combined/{halo_tag}/{model}.pt

    Parameters
    ----------
    halos : List[str]
        Halo model names (e.g., ['shm', 'shmpp', 'lmc']).
    modelname : str
        Model architecture name (e.g., 'hist_s1s2').
    n_train : int
        Number of training samples per halo.
    bins : int
        Number of S1/S2 bins per dimension.
    mu_bg : float
        Expected background events (0 = signal-only).
    training_mode : str
        'offline' or 'online'.
    device : str
        Device to load models onto.
    print_arch : bool
        Whether to print model architectures.

    Returns
    -------
    model_list : List[torch.nn.Module]
        [halo_1, halo_2, ..., combined] (combined only if len(halos) > 1).
    label_list : List[str]
        Corresponding labels.
    """
    if len(halos) < 1:
        raise ValueError("At least one halo must be specified.")

    signal_type = "signal_only" if mu_bg == 0 else f"signal_bg_mu{mu_bg:.0f}"
    model_list = []
    label_list = []

    for halo in halos:
        path = f"models/xenon/{training_mode}/{signal_type}/{halo}/{modelname}_bins{bins}_n{n_train}_{halo}.pt"
        print(f"Loading model from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        model, _ = load_model_s1s2(path, bins=bins, modelname=modelname, device=device, print_arch=print_arch)
        model_list.append(model)
        label_list.append(halo)

    if len(halos) > 1:
        halo_tag = "_".join(halos)
        combined_path = f"models/xenon/{training_mode}/{signal_type}/combined/{halo_tag}/{modelname}_bins{bins}_n{n_train}_{halo_tag}.pt"
        print(f"Loading combined model from {combined_path}")
        if not os.path.exists(combined_path):
            raise FileNotFoundError(f"Combined model not found at {combined_path}")
        model_combined, _ = load_model_s1s2(combined_path, bins=bins, modelname=modelname, device=device, print_arch=print_arch)
        model_list.append(model_combined)
        label_list.append("combined")

    return model_list, label_list

