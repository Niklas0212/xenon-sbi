"""
Train a neural network to infer WIMP parameters from simulated recoil spectra.

Supports:
- Single-halo training
- Multi-halo combined training

Usage examples:
    python3 -m sbi_training.training_wimpy --n_train 300000 --modelname ntothighest --datatag low --halos shm --batch_size 2048
    python3 -m sbi_training.training_wimpy --n_train 300000 --modelname full --datatag low --halos shm shmpp --batch_size 4096
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from configs.config import MCConfig, MODEL_CONFIG, TrainingDefaults, get_scheduler
from utils.processing import preprocess_features, make_dataloaders, make_negative_pairs
from utils.training import train


# ============================================================
# Configuration
# ============================================================

_TRAINING_DEFAULTS = TrainingDefaults()

@dataclass
class TrainingConfig:
    """Configuration for training."""
    top_k: int = _TRAINING_DEFAULTS.top_k
    batch_size_train: int = _TRAINING_DEFAULTS.batch_size_train
    batch_size_eval: int = _TRAINING_DEFAULTS.batch_size_eval
    num_workers: int = 4
    epochs: int = _TRAINING_DEFAULTS.epochs
    patience: int = _TRAINING_DEFAULTS.patience
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
        description="Train SBI model for WIMP parameter inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--n_train",
        type=int,
        default=300_000,
        help="Number of training samples per halo",
    )
    
    parser.add_argument(
        "--modelname",
        type=str,
        default="full",
        help="Model architecture to train",
    )
    
    parser.add_argument(
        "--datatag",
        type=str,
        default="low",
        choices=["low", "mid", "high"],
        help="Dataset tag indicating mass range",
    )
    
    parser.add_argument(
        "--halos",
        nargs="+",
        default=["default"],
        help="List of halo models to combine (e.g., default shm shmpp lmc)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=TrainingDefaults().batch_size_train,
        help="Batch size for train/validation/test dataloaders",
    )
    
    return parser.parse_args()


# ============================================================
# Helper functions
# ============================================================

def get_device_info(device: torch.device) -> str:
    """Get human-readable device information."""
    if device.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"


def get_model_path(
    n_train: int,
    modelname: str,
    datatag: str,
    halos: List[str],
) -> Tuple[Path, str]:
    """Determine model directory and path based on halo configuration."""
    halo_tag = "_".join(halos)
    
    if len(halos) > 1:
        model_dir = Path(f"models/wimpy/combined/{halo_tag}")
    else:
        model_dir = Path(f"models/wimpy/{halo_tag}")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{modelname}_n{n_train}_{datatag}_{halo_tag}.pt"
    
    return model_path, halo_tag


def load_datasets(
    halos: List[str],
    n_train: int,
    datatag: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and concatenate datasets from multiple halos."""
    feature_list = []
    theta_list = []
    
    print(f"Loading datasets for halos: {halos}")
    
    for halo in halos:
        datapath = Path(f"data/datasets/wimpy/{halo}/wimpy_n{n_train}_{datatag}_{halo}.pt")
        print(f"  → Loading {datapath}")
        
        if not datapath.exists():
            raise FileNotFoundError(f"Dataset not found: {datapath}")
        
        data = torch.load(datapath, weights_only=False)
        feature_list.append(data["features"])
        theta_list.append(data["theta"])
    
    # Concatenate along sample dimension
    feature_tensor = torch.cat(feature_list, dim=0)
    theta_tensor = torch.cat(theta_list, dim=0)
    
    print(f"Total combined training samples: {len(feature_tensor):,}")
    
    return feature_tensor, theta_tensor


def create_model(
    modelname: str,
    config: TrainingConfig,
) -> torch.nn.Module:
    """Create and configure model."""
    if modelname not in MODEL_CONFIG:
        raise ValueError(f"Unknown modelname: {modelname}. Available: {list(MODEL_CONFIG.keys())}")
    
    model_cfg = MODEL_CONFIG[modelname]
    input_dim = model_cfg["input_dim_fn"](config.bins, config.top_k)
    
    # Extract model arguments (exclude special keys)
    model_args = {
        k: v for k, v in model_cfg.items()
        if k not in {"cls", "optimizer", "input_dim_fn"}
    }
    
    model = model_cfg["cls"](input_dim=input_dim, **model_args).to(config.device)
    
    return model


def create_optimizer(model: torch.nn.Module, model_cfg: dict) -> torch.optim.Optimizer:
    """Create optimizer from model configuration."""
    opt_cfg = model_cfg["optimizer"]
    opt_type = opt_cfg["type"].lower()
    
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"])
    elif opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


# ============================================================
# Main
# ============================================================

def main():
    """Main training function."""
    # Parse arguments and initialize config
    args = parse_args()
    config = TrainingConfig()
    config.batch_size_train = args.batch_size
    config.batch_size_eval = args.batch_size
    
    # Device information
    print(f"Using device: {get_device_info(config.device)}")
    print(f"Training configuration:")
    print(f"  Model: {args.modelname}")
    print(f"  Data tag: {args.datatag}")
    print(f"  Samples per halo: {args.n_train:,}")
    print(f"  Halos: {args.halos}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {config.epochs} (patience: {config.patience})")
    
    # Determine model path
    model_path, halo_tag = get_model_path(
        args.n_train, args.modelname, args.datatag, args.halos
    )
    
    if model_path.exists():
        print(f"\nModel already exists at {model_path}. Skipping training.")
        return
    
    # Load and merge datasets
    feature_tensor, theta_tensor = load_datasets(
        args.halos, args.n_train, args.datatag
    )
    
    # Create model and optimizer
    print(f"\nInitializing model: {args.modelname}")
    model = create_model(args.modelname, config)
    model_cfg = MODEL_CONFIG[args.modelname]
    optimizer = create_optimizer(model, model_cfg)
    scheduler = get_scheduler(optimizer, model_cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Preprocess features
    print(f"\nPreprocessing features...")
    feature_tensor = preprocess_features(model, feature_tensor, config.device)
    
    # Create positive/negative pairs
    print(f"Creating positive/negative pairs...")
    X_all, T_all, Y_all = make_negative_pairs(feature_tensor, theta_tensor)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        X_all,
        T_all,
        Y_all,
        batch_size_train=config.batch_size_train,
        batch_size_eval=config.batch_size_eval,
        num_workers=config.num_workers,
    )
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Train model
    print(f"\nStarting training...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        epochs=config.epochs,
        patience=config.patience,
        ckpt_path=str(model_path),
    )
    
    print(f"\n✓ Training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
