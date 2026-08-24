"""
Training utilities for classifier-based Simulation-Based Inference (SBI).

Implements training loop with early stopping, checkpointing, and optional Optuna
integration for hyperparameter optimization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict
import optuna 
import matplotlib.pyplot as plt


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _accuracy_from_preds(preds: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute batch classification accuracy from raw predictions.
    
    Parameters
    ----------
    preds : torch.Tensor
        Raw model predictions (logits or probabilities).
    y_true : torch.Tensor
        True binary labels.
    
    Returns
    -------
    float
        Classification accuracy (fraction of correct predictions).
    """
    pred_labels = (preds >= 0.5).to(dtype=torch.long)
    return (pred_labels == y_true.to(dtype=torch.long)).float().mean().item()


# ============================================================================
# TRAINING LOOP
# ============================================================================


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 150,
    patience: int = 15,
    ckpt_path: Optional[str] = None,
    trial: Optional[optuna.trial.Trial] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    """
    Train neural network with early stopping and checkpointing.
    
    Implements binary cross-entropy training loop with validation-based early stopping.
    Tracks loss and accuracy for training, validation, and optionally test sets. 
    Optionally integrates with Optuna for hyperparameter optimization with trial pruning.
    
    Parameters
    ----------
    model : nn.Module
        Neural network model to train.
    train_loader : DataLoader
        DataLoader for training data. Must provide (features, theta, labels) tuples.
    val_loader : DataLoader
        DataLoader for validation data.
    test_loader : Optional[DataLoader], optional
        DataLoader for test data. If provided, test metrics are tracked each epoch.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    device : torch.device
        Device for computation ('cpu' or 'cuda').
    epochs : int, optional
        Maximum number of training epochs (default: 150).
    patience : int, optional
        Number of epochs without improvement before early stopping (default: 15).
    ckpt_path : Optional[str], optional
        Path to save best model checkpoint. If None, no checkpoint is saved.
    trial : Optional[optuna.trial.Trial], optional
        Optuna trial object for hyperparameter optimization. If provided, reports
        validation loss and enables pruning.
    scheduler : Optional[torch.optim.lr_scheduler.LRScheduler], optional
        Learning rate scheduler stepped once per epoch. If None, learning rate
        remains constant.
    
    Returns
    -------
    train_loss_history : list[float]
        Training loss per epoch.
    val_loss_history : list[float]
        Validation loss per epoch.
    train_acc_history : list[float]
        Training accuracy per epoch.
    val_acc_history : list[float]
        Validation accuracy per epoch.
    test_loss_history : list[float]
        Test loss per epoch (empty if test_loader not provided).
    test_acc_history : list[float]
        Test accuracy per epoch (empty if test_loader not provided).
    
    Notes
    -----
    - Model state is restored to best validation loss before returning
    - Checkpoint includes model state, loss/accuracy histories, and best metrics
    - Early stopping compares with 1e-6 tolerance to avoid numerical noise
    - Test set is only used for tracking, not for early stopping decisions
    """

    best_loss, best_accuracy = float("inf"), 0.0
    best_state = None
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    test_loss_history, test_acc_history = [], []
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):

        # ========================================
        # TRAINING PHASE
        # ========================================
        model.train()
        running_loss, running_acc, total_batches = 0.0, 0.0, 0
        for xb, tb, yb in train_loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            preds = model(xb, tb).squeeze(-1).clamp(1e-10, 1 - 1e-10)

            # Compute loss and backpropagate
            loss = F.binary_cross_entropy(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_loss += loss.item()
            running_acc += _accuracy_from_preds(preds, yb)
            total_batches += 1

        train_loss = running_loss / total_batches
        train_acc = running_acc / total_batches
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)


        # ========================================
        # VALIDATION PHASE
        # ========================================
        model.eval()
        running_loss, running_acc, total_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, tb, yb in val_loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                preds = model(xb, tb).squeeze(-1).clamp(1e-10, 1 - 1e-10)
                loss = F.binary_cross_entropy(preds, yb)

                # Accumulate metrics
                running_loss += loss.item()
                running_acc += _accuracy_from_preds(preds, yb)
                total_batches += 1

        val_loss = running_loss / total_batches
        val_acc = running_acc / total_batches
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # ========================================
        # TEST PHASE (if test_loader provided)
        # ========================================
        if test_loader is not None:
            running_loss, running_acc, total_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, tb, yb in test_loader:
                    xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                    preds = model(xb, tb).squeeze(-1).clamp(1e-10, 1 - 1e-10)
                    loss = F.binary_cross_entropy(preds, yb)

                    # Accumulate metrics
                    running_loss += loss.item()
                    running_acc += _accuracy_from_preds(preds, yb)
                    total_batches += 1

            test_loss = running_loss / total_batches
            test_acc = running_acc / total_batches
            test_loss_history.append(test_loss)
            test_acc_history.append(test_acc)
            

        # ========================================
        #  SCHEDULER STEP
        # ========================================
        if scheduler is not None:
            scheduler.step()


        # ========================================
        # LOGGING
        # ========================================
        # Print header on first epoch (if not using Optuna)
        if trial is None and epoch == 1:
            print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Train Acc':>9} | {'Val Acc':>8}")
            print("-" * 60)

        # Print progress
        if trial is None:
            print(f"{epoch:5d} | {train_loss:10.3f} | {val_loss:8.3f} | {train_acc:9.3f} | {val_acc:8.3f}")
       

        # ========================================
        # OPTUNA INTEGRATION
        # ========================================
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            

        # ========================================
        # EARLY STOPPING
        # ========================================
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_accuracy = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


    # ========================================
    # RESTORE BEST MODEL AND SAVE CHECKPOINT
    # ========================================
    if best_state is not None:
        model.load_state_dict(best_state)
            
    if ckpt_path is not None:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "train_acc": train_acc_history,
            "val_acc": val_acc_history,
            "best_val_loss": best_loss,
            "best_val_acc": best_accuracy,
        }
        
        # Add test metrics if available
        if test_loader is not None:
            checkpoint_data["test_loss"] = test_loss_history
            checkpoint_data["test_acc"] = test_acc_history
        
        torch.save(checkpoint_data, ckpt_path)
        print(f"\nSaved checkpoint to {ckpt_path}")
    
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history, test_loss_history, test_acc_history




# ============================================================================
# TRAINING ANALYSIS
# ============================================================================


def plot_training_summary(
    ckpt: Dict,
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    title_fontsize: int = 26,
    label_fontsize: int = 24,
    tick_fontsize: int = 22,
    legend_fontsize: int = 21,
    line_width: float = 5.0,
    alpha_train: float = 0.95,
    alpha_val: float = 0.95,
    alpha_test: float = 1.00,
):
    """Print training summary and plot loss/accuracy curves from a checkpoint.

    Parameters
    ----------
    ckpt : dict
        Checkpoint dictionary (as saved by ``train``). Expected keys:
        ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``, and
        optionally ``best_val_loss``, ``best_val_acc``, ``test_loss``,
        ``test_acc``.
    figsize : tuple
        Figure size for the two-panel plot.
    save_path : Optional[str]
        Optional file path to save the figure (e.g. "figures/training_summary.pdf").
        If None, the figure is not saved.
    dpi : int
        Figure DPI used when saving.
    title_fontsize : int
        Font size for subplot titles.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for axis ticks.
    legend_fontsize : int
        Font size for legend text.
    line_width : float
        Line width used for all curves.
    alpha_train : float
        Opacity for train curves.
    alpha_val : float
        Opacity for validation curves.
    alpha_test : float
        Opacity for test curves.
    """
    train_losses = ckpt.get("train_loss", [])
    val_losses   = ckpt.get("val_loss", [])
    train_acc    = ckpt.get("train_acc", [])
    val_acc      = ckpt.get("val_acc", [])
    test_losses  = ckpt.get("test_loss", [])
    test_acc     = ckpt.get("test_acc", [])

    best_val_loss = ckpt.get("best_val_loss", min(val_losses) if val_losses else None)
    best_val_acc  = ckpt.get("best_val_acc",  max(val_acc)   if val_acc   else None)

    # ---- Summary ----
    print("\nTraining Summary")
    print("────────────────────────────────────────────")
    if best_val_loss is not None:
        print(f"Best validation loss:     {best_val_loss:.3f}")
    if best_val_acc is not None:
        print(f"Best validation accuracy: {best_val_acc:.3f}")
    print("────────────────────────────────────────────")

    # ---- Plotting ----
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

    train_color = "#1f77b4"
    val_color = "#ff7f0e"
    test_color = "darkgray"

    # Loss
    axes[0].plot(train_losses, label="Train", linewidth=line_width, color=train_color, alpha=alpha_train)
    axes[0].plot(val_losses, label="Validation", linewidth=line_width, color=val_color, alpha=alpha_val)
    if test_losses:
        axes[0].plot(
            test_losses,
            label="Test",
            linewidth=line_width,
            color=test_color,
            alpha=alpha_test,
            linestyle="-",
        )
    axes[0].set_title("Loss Curve", fontsize=title_fontsize)
    axes[0].set_xlabel("Epoch", fontsize=label_fontsize)
    axes[0].set_ylabel("BCE Loss", fontsize=label_fontsize)
    axes[0].tick_params(axis="both", labelsize=tick_fontsize)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    # Accuracy
    axes[1].plot(train_acc, label="Train", linewidth=line_width, color=train_color, alpha=alpha_train)
    axes[1].plot(val_acc, label="Validation", linewidth=line_width, color=val_color, alpha=alpha_val)
    if test_acc:
        axes[1].plot(
            test_acc,
            label="Test",
            linewidth=line_width,
            color=test_color,
            alpha=alpha_test,
            linestyle="-",
        )
    axes[1].set_title("Accuracy Curve", fontsize=title_fontsize)
    axes[1].set_xlabel("Epoch", fontsize=label_fontsize)
    axes[1].set_ylabel("Accuracy", fontsize=label_fontsize)
    axes[1].tick_params(axis="both", labelsize=tick_fontsize)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=legend_fontsize)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved training summary figure to {save_path}")

    plt.show()