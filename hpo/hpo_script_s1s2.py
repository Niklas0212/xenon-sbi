"""
Run Optuna hyperparameter optimization for S1-S2 classifier models.

Approach:
- Load signal/background data once
- Build fixed histograms for each trial
- Run exactly one study mode per execution:
  - signal_only if bg=False
  - signal_bg if bg=True (with fixed mu_bg from HPOConfig)

Usage:
	python3 -m hpo.hpo_script_s1s2 False --n 100
	python3 -m hpo.hpo_script_s1s2 True --n 100
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from configs.config import MODEL_CONFIG
from utils.processing import (
	SignalPlusBackgroundDataset,
	clean_nan_events,
	make_negative_pairs,
)
from utils.training import train


# ============================================================
# Configuration
# ============================================================

@dataclass
class HPOConfig:
	"""Configuration for S1S2 hyperparameter optimization."""
	n_train: int = 300_000
	halo: str = "default"
	modelname: str = "hist_s1s2"
	epochs: int = 150
	patience: int = 20
	num_workers: int = 0
	device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	bin_choices: Tuple[int, ...] = (10, 15, 20, 30, 40, 50)
	mu_bg_signal: float = 150.0


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
	"""Parse command line arguments."""
	def _to_bool(value: str) -> bool:
		if isinstance(value, bool):
			return value
		value_lower = value.strip().lower()
		if value_lower in {"true", "1", "yes", "y"}:
			return True
		if value_lower in {"false", "0", "no", "n"}:
			return False
		raise argparse.ArgumentTypeError("Use true/false for --bg")

	parser = argparse.ArgumentParser(
		description="Optuna HPO for S1S2 model",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument("--n", type=int, default=100, help="Number of optimization trials per study")
	parser.add_argument("bg", type=_to_bool, help="True runs signal_bg study, False runs signal_only")

	return parser.parse_args()


# ============================================================
# Data loading
# ============================================================

def load_base_data(config: HPOConfig):
	"""Load and prepare signal/background data."""
	signal_file = f"data/datasets/xenon/s1s2/pt/s1s2_n{config.n_train}_{config.halo}.pt"
	if not os.path.exists(signal_file):
		raise FileNotFoundError(f"Signal file not found: {signal_file}")

	print(f"Loading signal from {signal_file}")
	data = torch.load(signal_file, weights_only=False)
	energies_list, cs1cs2_list, theta = data["events"], data["cs1cs2"], data["theta"]
	_, signal_events = clean_nan_events(energies_list, cs1cs2_list)

	signal_train, signal_val, theta_train, theta_val = train_test_split(
		signal_events,
		theta,
		test_size=0.20,
		random_state=42,
		shuffle=True,
	)

	bg_file = "data/datasets/xenon/s1s2/ers/s1s2_ers.csv"
	if not os.path.exists(bg_file):
		raise FileNotFoundError(f"Background file not found: {bg_file}")

	background_events = np.loadtxt(bg_file, delimiter=",", skiprows=1, usecols=(0, 1))

	print(f"  Combined spectra: {len(theta):,}")
	print(f"  Train spectra: {len(signal_train):,}")
	print(f"  Val spectra: {len(signal_val):,}")
	print(f"  Background pool: {len(background_events):,} events")

	return signal_train, signal_val, theta_train, theta_val, background_events


def _build_loader_from_dataset(dataset: SignalPlusBackgroundDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
	"""Generate fixed histograms for one dataset split and wrap in DataLoader."""
	x_list, t_list = [], []
	for x, t in dataset:
		x_list.append(x.view(-1))
		t_list.append(t)

	x_all = torch.stack(x_list)
	t_all = torch.stack(t_list)
	x_pairs, t_pairs, y_pairs = make_negative_pairs(x_all, t_all)

	tensor_dataset = TensorDataset(x_pairs, t_pairs, y_pairs)
	return DataLoader(
		tensor_dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
	)


def prepare_dataloaders(
	signal_train,
	signal_val,
	theta_train: torch.Tensor,
	theta_val: torch.Tensor,
	background_events,
	bins: int,
	mu_bg: float,
	batch_size: int,
	num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
	"""Create train/val dataloaders for given (bins, mu_bg)."""
	train_dataset = SignalPlusBackgroundDataset(
		signal_events=signal_train,
		theta=theta_train,
		background_events=background_events,
		mu_bg=mu_bg,
		s1_bins=bins,
		s2_bins=bins,
		reproducible=False,
	)

	val_dataset = SignalPlusBackgroundDataset(
		signal_events=signal_val,
		theta=theta_val,
		background_events=background_events,
		mu_bg=mu_bg,
		s1_bins=bins,
		s2_bins=bins,
		reproducible=True,
		rng_seed=42,
	)

	train_loader = _build_loader_from_dataset(train_dataset, batch_size, num_workers, shuffle=True)
	val_loader = _build_loader_from_dataset(val_dataset, batch_size, num_workers, shuffle=False)
	return train_loader, val_loader


# ============================================================
# Optuna objective
# ============================================================

def create_model(trial: optuna.Trial, bins: int, config: HPOConfig) -> torch.nn.Module:
	"""Create S1S2 model using trial hyperparameters."""
	hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
	num_layers = trial.suggest_int("num_layers", 4, 8)
	dropout = trial.suggest_categorical("dropout", [0.0, 0.03, 0.05, 0.1, 0.15, 0.2])

	model_cfg = MODEL_CONFIG[config.modelname]
	input_dim = model_cfg["input_dim_fn"](bins, bins)
	return model_cfg["cls"](
		input_dim=input_dim,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		dropout=dropout,
		batchnorm=True,
	).to(config.device)


def create_optimizer(trial: optuna.Trial, model: torch.nn.Module) -> torch.optim.Optimizer:
	"""Create optimizer from trial suggestions."""
	lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2])
	weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

	return torch.optim.AdamW(
		model.parameters(),
		lr=lr,
		weight_decay=weight_decay,
	)


def objective(
	trial: optuna.Trial,
	config: HPOConfig,
	signal_train,
	signal_val,
	theta_train: torch.Tensor,
	theta_val: torch.Tensor,
	background_events,
	mu_bg: float,
) -> float:
	"""Optuna objective for one S1S2 scenario (signal_only or signal_bg)."""
	bins = trial.suggest_categorical("bins", list(config.bin_choices))
	batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
	lr_decay = trial.suggest_categorical("lr_decay", [0.95, 0.97, 0.99])

	model = create_model(trial, bins, config)
	optimizer = create_optimizer(trial, model)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

	train_loader, val_loader = prepare_dataloaders(
		signal_train=signal_train,
		signal_val=signal_val,
		theta_train=theta_train,
		theta_val=theta_val,
		background_events=background_events,
		bins=bins,
		mu_bg=mu_bg,
		batch_size=batch_size,
		num_workers=config.num_workers,
	)

	_, val_losses, _, _, _, _ = train(
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


def print_study_summary(study: optuna.Study, label: str) -> None:
	"""Print concise study summary."""
	print(f"\nStudy summary ({label}):")
	print(f"  Finished trials: {len(study.trials)}")
	print(f"  Best value: {study.best_value:.6f}")
	print("  Best params:")
	for key, value in study.best_params.items():
		print(f"    {key}: {value}")


def run_study(
	label: str,
	mu_bg: float,
	n_trials: int,
	n_jobs: int,
	config: HPOConfig,
	signal_train,
	signal_val,
	theta_train: torch.Tensor,
	theta_val: torch.Tensor,
	background_events,
) -> None:
	"""Create/load and run one Optuna study."""
	study_dir = Path("hpo/optuna_studies")
	study_dir.mkdir(parents=True, exist_ok=True)

	study_name = f"{config.modelname}__{label}"
	storage = f"sqlite:///{study_dir}/{study_name}.db"

	print(f"\nInitializing study: {study_name}")
	print(f"  Scenario: {label}, mu_bg={mu_bg}")

	study = optuna.create_study(
		study_name=study_name,
		storage=storage,
		load_if_exists=True,
		direction="minimize",
		sampler=optuna.samplers.TPESampler(),
		pruner=optuna.pruners.MedianPruner(),
	)

	print(f"  Running with {n_jobs} parallel jobs")
	print(f"  Target trials: {n_trials}")

	study.optimize(
		lambda trial: objective(
			trial=trial,
			config=config,
			signal_train=signal_train,
			signal_val=signal_val,
			theta_train=theta_train,
			theta_val=theta_val,
			background_events=background_events,
			mu_bg=mu_bg,
		),
		n_trials=n_trials,
		n_jobs=n_jobs,
	)

	print_study_summary(study, label)


# ============================================================
# Main
# ============================================================

def main() -> None:
	"""Main entry point."""
	args = parse_args()
	config = HPOConfig()
	n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

	label = "signal_bg" if args.bg else "signal_only"
	mu_bg = config.mu_bg_signal if args.bg else 0.0

	print("Loading S1S2 data")
	print(f"  Model: {config.modelname}")
	print(f"  Halo: {config.halo}")
	print(f"  n_train per halo: {config.n_train}")
	print(f"  Device: {config.device}")
	print(f"  Study mode: {label}")
	print(f"  Running with {n_jobs} parallel jobs")
	print(f"  Target trials: {args.n}")

	signal_train, signal_val, theta_train, theta_val, background_events = load_base_data(config)

	run_study(
		label=label,
		mu_bg=mu_bg,
		n_trials=args.n,
		n_jobs=n_jobs,
		config=config,
		signal_train=signal_train,
		signal_val=signal_val,
		theta_train=theta_train,
		theta_val=theta_val,
		background_events=background_events,
	)


if __name__ == "__main__":
	main()

