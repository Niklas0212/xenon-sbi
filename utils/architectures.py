"""
Neural network architectures for simulation-based inference of WIMP parameters.

"""

import torch
import torch.nn as nn

# ================================================================
# UNIVERSAL MLP CLASS (core logic used by all models)
# ================================================================

class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4,
                 dropout=0.05, batchnorm=True):
        super().__init__()

        dim_in = input_dim + 2  # (x + theta)

        layers = []

        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim_in = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x, theta):
        return self.net(torch.cat([x, theta], dim=-1))


# ================================================================
# MLP ARCHITECTURES (thin wrappers, just for naming)
# ================================================================

class Full_MLP(BaseMLP):
    """Hist + Ntot + Highest"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)

class Ntot_Highest_MLP(BaseMLP):
    """Ntot + Highest"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)

class Ntot_Highest_MLP_Vanilla(BaseMLP):
    """Ntot + Highest, without hpo"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)

class Hist_MLP(BaseMLP):
    """Only Hist"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)

class Ntot_MLP(BaseMLP):
    """Only Ntot"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)

class Highest_MLP(BaseMLP):
    """Only top-k energies"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)


# ================================================================
# XENON ARCHITECTURES (thin wrappers, just for naming)
# ================================================================

class HistS1S2_MLP(BaseMLP):
    """Flattened 2d S1-S2 histograms"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)


class S1S2_signal(BaseMLP):
    """Flattened 2d S1-S2 histograms (signal-only tuned model)."""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)


class S1S2_signal_bg(BaseMLP):
    """Flattened 2d S1-S2 histograms (signal+background tuned model)."""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)






