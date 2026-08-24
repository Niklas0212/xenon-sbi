"""
Train a classifier NN to infer WIMP parameters from S1-S2 histograms.

General approach:
- Generate all histograms ONCE (with sampled backgrounds)
- Create fixed tensors
- Train on fixed tensors (fast, reproducible)

Supports:
- Signal only (mu_bg=0)
- Signal + Background (mu_bg > 0)
- Single-halo training
- Multi-halo combined training
- Automatic model selection:
    - mu_bg=0   -> S1S2_signal
    - mu_bg>0   -> S1S2_signal_bg

Workflow:
---------
1. LOAD: Signal spectra (.pt file) + Background events (.csv file)
2. CLEAN: Remove NaN events using clean_nan_events()
3. SPLIT: Train/Test split (80/20) on SIGNAL spectra only
4. CREATE DATASETS: SignalPlusBackgroundDataset[i] returns (histogram, theta)
5. GENERATE HISTOGRAMS & PAIRS: Create fixed tensors with sampled backgrounds
6. TRAIN: Classifier NN on fixed train/test tensors

Usage example (general argparse form):
    python3 -m sbi_training.training_s1s2 --halos shm     --mu_bg 0   --bins 10 --batch_size 1024
    python3 -m sbi_training.training_s1s2 --halos shm shmpp --mu_bg 150 --bins 10 --batch_size 1024
"""


import os
import sys
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset

# Add parent dir to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from configs.config import MODEL_CONFIG, TrainingDefaults, get_optimizer, get_scheduler
from utils.processing import make_negative_pairs, clean_nan_events, SignalPlusBackgroundDataset
from utils.training import train


def load_datasets(n_train, halos):

    # Load and merge signal events across halos
    energies_all = []
    cs1cs2_all = []
    theta_all = []

    for h in halos:
        signal_file = f"data/datasets/xenon/s1s2/pt/s1s2_n{n_train}_{h}.pt"
        print(f"Loading signal from {signal_file}")

        if not os.path.exists(signal_file):
            raise FileNotFoundError(f"Signal file not found: {signal_file}")

        data = torch.load(signal_file, weights_only=False)
        energies_list = data["events"]
        cs1cs2_list = data["cs1cs2"]
        theta = data["theta"]
        print(f"  → Loaded {len(theta):,} signal spectra (before cleaning)")

        energies_list, cs1cs2_list = clean_nan_events(energies_list, cs1cs2_list)

        # energies_list/cs1cs2_list are typically lists; merge accordingly
        if torch.is_tensor(energies_list):
            energies_all.append(energies_list)
        else:
            energies_all.extend(energies_list)

        if torch.is_tensor(cs1cs2_list):
            cs1cs2_all.append(cs1cs2_list)
        else:
            cs1cs2_all.extend(cs1cs2_list)

        theta_all.append(theta)

    energies_list = torch.cat(energies_all, dim=0) if energies_all and torch.is_tensor(energies_all[0]) else energies_all
    cs1cs2_list = torch.cat(cs1cs2_all, dim=0) if cs1cs2_all and torch.is_tensor(cs1cs2_all[0]) else cs1cs2_all
    theta = torch.cat(theta_all, dim=0)

    print(f"  → Combined {len(theta):,} signal spectra (after cleaning)")

    # Train/Test Split (80/20)
    print("\nSplitting into train (80%) and test (20%) sets...")
    signal_events_train, signal_events_test, theta_train, theta_test = train_test_split(
        cs1cs2_list, theta,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )
    print(f"  → Train: {len(signal_events_train):,} spectra")
    print(f"  → Test:  {len(signal_events_test):,} spectra")

    # Load background events
    bg_file = "data/datasets/xenon/s1s2/ers/s1s2_ers.csv"
    print(f"\nLoading background from {bg_file}")
    
    if not os.path.exists(bg_file): raise FileNotFoundError(f"Background file not found: {bg_file}")
        
    background_events = np.loadtxt(bg_file, delimiter=",", skiprows=1, usecols=(0, 1))
    print(f"  → Loaded {len(background_events):,} background events")

    return signal_events_train, signal_events_test, theta_train, theta_test, background_events


def generate_histograms_and_dataloaders(dataset, batch_size, label):
    """
    Generate fixed histograms from dataset, create pairs, and return DataLoader.
    
    Args:
        dataset: SignalPlusBackgroundDataset instance
        batch_size: Batch size for DataLoader
        label: String label ("train" or "test") for printing
    
    Returns:
        DataLoader wrapping TensorDataset(X, T, Y)
    """
    print(f"\nGenerating {label} histograms...")
    X, T = [], []
    for x, t in dataset:
        X.append(x.view(-1)) # Flatten histogram to 1D
        T.append(t)
    X = torch.stack(X)
    T = torch.stack(T)
    print(f"  → Generated {len(X):,} histograms")
    
    print(f"Creating positive/negative pairs for {label}...")
    X, T, Y = make_negative_pairs(X, T)
    print(f"  → Created {len(X):,} pairs (including negatives)")
    
    # Create tensor dataset and dataloader
    tensor_dataset = TensorDataset(X, T, Y)
    loader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    return loader


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defaults = TrainingDefaults()

    # Arguments
    parser = argparse.ArgumentParser(description="Train classifier NN with signal+background")
    parser.add_argument("--n_train", type=int, default=300_000, help="Number of training samples per halo")
    parser.add_argument(
        "--halos",
        nargs="+",
        default=["shm"],
        help="List of halo models to combine (e.g., shm shmpp lmc)",
    )
    parser.add_argument("--mu_bg", type=float, default=150, help="Expected number of background events (0 = signal only)")
    parser.add_argument("--bins", type=int, default=defaults.s1s2_bins, help="Number of S1 and S2 bins")
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size_train, help="Batch size")
    parser.add_argument("--patience", type=int, default=defaults.patience, help="Early stopping patience")
    args = parser.parse_args()

    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
    print(f"\nTraining configuration:")
    modelname = "S1S2_signal" if args.mu_bg == 0 else "S1S2_signal_bg"

    print(f"  Model: {modelname}")
    print(f"  Samples per halo: {args.n_train:,}")
    print(f"  Halos: {args.halos}")
    print(f"  Expected background events: {args.mu_bg}")
    print(f"  Bins: {args.bins}x{args.bins}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs} (patience: {args.patience})")

    n_train = args.n_train
    halos = args.halos
    mu_bg = args.mu_bg
    bins = args.bins

    if modelname not in MODEL_CONFIG:
        raise ValueError(f"Unknown modelname: {modelname}")

    cfg = MODEL_CONFIG[modelname]

    # Determine halo tag and directory structure (consistent with training_wimpy.py)
    # Single halo: models/xenon/{signal_type}/{halo}/{modelname}.pt
    # Multi halo : models/xenon/{signal_type}/combined/{halo}/{modelname}.pt
    halo_tag = "_".join(halos)
    signal_type = "signal_only" if mu_bg == 0 else f"signal_bg_mu{mu_bg:.0f}"
    
    if len(halos) > 1:
        MODELDIR = f"models/xenon/{signal_type}/combined/{halo_tag}"
    else:
        MODELDIR = f"models/xenon/{signal_type}/{halo_tag}"
    
    os.makedirs(MODELDIR, exist_ok=True)
    MODELPATH = os.path.join(MODELDIR, f"{modelname}_bins{bins}_n{n_train}_{halo_tag}.pt")

    if os.path.exists(MODELPATH):
        print(f"Model already exists at {MODELPATH}. Skipping training.")
        return

    # ========================================================================
    # Load datasets
    # ========================================================================
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    signal_events_train, signal_events_test, theta_train, theta_test, background_events = load_datasets(n_train, halos)

    # Training dataset
    train_dataset = SignalPlusBackgroundDataset(
        signal_events=signal_events_train,
        theta=theta_train,
        background_events=background_events,
        mu_bg=mu_bg,
        s1_bins=bins,
        s2_bins=bins,
    )

    # Test dataset
    test_dataset = SignalPlusBackgroundDataset(
        signal_events=signal_events_test,
        theta=theta_test,
        background_events=background_events,
        mu_bg=mu_bg,
        s1_bins=bins,
        s2_bins=bins,
        reproducible=True,
        rng_seed=42
    )

    # ========================================================================
    # Generate histograms and dataloaders
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING HISTOGRAMS AND PAIRS")
    print("="*70)
    
    train_loader = generate_histograms_and_dataloaders(train_dataset, args.batch_size, "train")
    test_loader = generate_histograms_and_dataloaders(test_dataset, args.batch_size, "test")

    # ========================================================================
    # Initialize model
    # ========================================================================
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    input_dim = cfg["input_dim_fn"](bins, bins)
    model_args = {k: v for k, v in cfg.items() if k not in {"cls", "optimizer", "input_dim_fn"}}
    model = cfg["cls"](input_dim=input_dim, **model_args).to(device)
    
    print(f"Model: {modelname}")
    print(f"Input dimension: {input_dim} ({bins}x{bins})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    print(f"Optimizer: {cfg['optimizer']}")

    # ========================================================================
    # Train
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    print(f"Training samples: {len(train_loader) * args.batch_size:,}")
    print(f"Test samples: {len(test_loader) * args.batch_size:,}")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}")
    print("="*70 + "\n")

    train(
        model, 
        train_loader, 
        test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs, 
        patience=args.patience, 
        ckpt_path=MODELPATH
    )

    print(f"\nSaved model to: {MODELPATH}")


if __name__ == "__main__":
    main()
