"""
Run Optuna hyperparameter optimization for SBI classifier models.
Run: python3 -m hpo.hpo_script --datatag low --modelname ntothighest --n 200 
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import optuna
from torch.utils.data import DataLoader

from configs.config import MCConfig, MODEL_CONFIG
from utils.processing import (
    preprocess_features,
    make_dataloaders,
    make_negative_pairs,
)
from utils.training import train


# ============================================================
# Configuration
# ============================================================

@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    n_train: int = 300_000
    halo: str = "default"
    top_k: int = 10
    epochs: int = 150
    patience: int = 20
    num_workers: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def bins(self):
        """Get bins configuration from MCConfig."""
        return MCConfig().bins


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna HPO for SBI models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--datatag",
        type=str,
        required=True,
        choices=["low", "mid", "high"],
        help="Dataset tag indicating mass range",
    )

    parser.add_argument(
        "--modelname",
        type=str,
        required=True,
        choices=["full", "ntothighest"],
        help="Model architecture to optimize",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of optimization trials",
    )

    return parser.parse_args()


# ============================================================
# Helper functions
# ============================================================

def load_data(datatag: str, config: HPOConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load training data from disk."""
    datapath = Path(f"data/datasets/wimpy/{config.halo}/wimpy_n{config.n_train}_{datatag}_{config.halo}.pt")
    data = torch.load(datapath, weights_only=False)
    return data["features"], data["theta"]


def create_model(trial: optuna.Trial, modelname: str, config: HPOConfig) -> torch.nn.Module:
    """Create and configure model based on trial suggestions."""
    # Suggest architecture hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 4, 8)
    dropout = trial.suggest_categorical("dropout", [0.0, 0.03, 0.05, 0.1, 0.15, 0.2])
    
    # Get model configuration and create model
    model_cfg = MODEL_CONFIG[modelname]
    input_dim = model_cfg["input_dim_fn"](config.bins, config.top_k)
    model = model_cfg["cls"](
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        batchnorm=True,
    ).to(config.device)
    
    return model


def create_optimizer(trial: optuna.Trial, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create optimizer based on trial suggestions."""
    lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def prepare_dataloaders(
    model: torch.nn.Module,
    feature_tensor: torch.Tensor,
    theta_tensor: torch.Tensor,
    config: HPOConfig,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation dataloaders."""
    features_proc = preprocess_features(model, feature_tensor, config.device)
    X_all, T_all, Y_all = make_negative_pairs(features_proc, theta_tensor)
    
    train_loader, val_loader, _ = make_dataloaders(
        X_all,
        T_all,
        Y_all,
        batch_size_train=batch_size,
        batch_size_eval=batch_size,
        num_workers=config.num_workers,
    )
    
    return train_loader, val_loader


# ============================================================
# Optuna objective
# ============================================================

def objective(
    trial: optuna.Trial,
    modelname: str,
    feature_tensor: torch.Tensor,
    theta_tensor: torch.Tensor,
    config: HPOConfig,
) -> float:
    """Objective function for Optuna optimization."""
    # Create model and optimizer
    model = create_model(trial, modelname, config)
    optimizer = create_optimizer(trial, model)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    lr_decay = trial.suggest_categorical("lr_decay", [0.95, 0.97, 0.99])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    # Prepare data
    train_loader, val_loader = prepare_dataloaders(
        model, feature_tensor, theta_tensor, config, batch_size
    )
    
    # Train model
    _, val_losses, _, _, _, _= train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        epochs=config.epochs,
        patience=config.patience,
        ckpt_path=None,
        trial=trial,
    )
    
    return min(val_losses)


def print_study_summary(study: optuna.Study) -> None:
    """Print summary of optimization study."""
    print("\nStudy statistics:")
    print(f"  Finished trials: {len(study.trials)}")
    print(f"  Best value: {study.best_value:.6f}")
    print("  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")


# ============================================================
# Main
# ============================================================

def main():
    """Main execution function."""
    # Parse arguments and initialize config
    args = parse_args()
    config = HPOConfig()
    
    # Load data
    print(f"Loading data: {args.datatag} mass range, {args.modelname} model")
    feature_tensor, theta_tensor = load_data(args.datatag, config)
    print(f"  Features shape: {feature_tensor.shape}")
    print(f"  Theta shape: {theta_tensor.shape}")
    print(f"  Device: {config.device}")
    
    # Setup study storage
    study_dir = Path("hpo/optuna_studies")
    study_dir.mkdir(parents=True, exist_ok=True)
    

    storage = f"sqlite:///{study_dir}/{args.modelname}_{args.datatag}.db"
    study_name = f"{args.modelname}_{args.datatag}"
    
    # Create or load study
    print(f"\nInitializing Optuna study: {study_name}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    
    # Determine parallelization
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"  Running with {n_jobs} parallel jobs")
    print(f"  Target trials: {args.n}")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            args.modelname,
            feature_tensor,
            theta_tensor,
            config,
        ),
        n_trials=args.n,
        n_jobs=n_jobs,
    )
    
    # Print results
    print_study_summary(study)


if __name__ == "__main__":
    main()
