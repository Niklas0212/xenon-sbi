"""
Data preprocessing and dataloader utilities for Simulation-Based Inference (SBI).

Provides functions for feature preprocessing, train/val/test splitting, data loading,
and dataset utilities for both WIMP spectrum and S1S2 histogram models.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ============================================================================
# FEATURE PREPROCESSING
# ============================================================================

def preprocess_features(model: nn.Module, counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert raw simulation output into the correct feature representation
    for a given model type.

    Models are defined by which feature indices they use:
      - hist  : 0-99
      - ntot  : index 100
      - highE : indices >= 101
    """

    name = model.__class__.__name__

    # model-to-feature mapping
    MODEL_FEATURE_MAP = {
        "Full_MLP"                  : slice(None),
        "Ntot_Highest_MLP"          : slice(100, None),
        "Ntot_Highest_MLP_Vanilla"  : slice(100, None),
        "Hist_MLP"                  : slice(0, 100),
        "Ntot_MLP"                  : 100, # scalar → keep dim later
        "Highest_MLP"               : slice(101, None),
        "HistS1S2_MLP"              : slice(None),  # Use full S1S2 histogram (flattened before)
        "S1S2_signal"               : slice(None),
        "S1S2_signal_bg"            : slice(None),
    }

    if name not in MODEL_FEATURE_MAP:
        raise ValueError(f"Unknown model type '{name}' in preprocess_features")

    idx = MODEL_FEATURE_MAP[name]

    # --- handle batch & non-batch cases uniformly ---
    counts = torch.as_tensor(counts, dtype=torch.float32, device=device)

    if isinstance(idx, int):               # models with only one scalar feature (e.g. NtotMLP)
        feats = counts[..., idx]
        if counts.ndim == 2:
            feats = feats.unsqueeze(1)     # shape → (B,1)
    else:
        feats = counts[..., idx]           # slice or list → multi-dim features

    if name not in {"HistS1S2_MLP", "S1S2_signal", "S1S2_signal_bg"}:
        feats = torch.log10(feats + 1)     # log transform for wimpy models only 

    return feats.float().to(device)


# ============================================================================
# DATA SPLITTING AND LOADING
# ============================================================================

def make_negative_pairs(features, thetas):
    """Create non-matching (x, theta) pairs by permuting features."""
    perm = torch.randperm(len(features), generator=torch.Generator().manual_seed(42))
    X = torch.cat([features, features[perm]])
    T = torch.cat([thetas,  thetas])
    Y = torch.cat([torch.ones(len(features)), torch.zeros(len(features))])
    return X, T, Y


def make_dataloaders(
    features_all: torch.Tensor,
    thetas_all: torch.Tensor,
    labels_all: torch.Tensor,
    batch_size_train: int = 4096,
    batch_size_eval: int = 4096,
    val_size: float = 0.15,
    test_size: float = 0.05,
    num_workers: int = 0,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders with reproducible splits.
    
    This function uses a fixed random_state to ensure reproducibility. When called
    with the same inputs and random_state, it will produce identical train/val/test
    splits. To get matching pairs from these splits later, use get_matching_pairs().
    
    Parameters
    ----------
    features_all : torch.Tensor
        All feature vectors.
    thetas_all : torch.Tensor
        All parameter vectors.
    labels_all : torch.Tensor
        All labels (1 for matching pairs, 0 for negative pairs).
    batch_size_train : int
        Batch size for training loader.
    batch_size_eval : int
        Batch size for validation and test loaders.
    val_size : float
        Fraction of data for validation (of total dataset).
    test_size : float
        Fraction of data for test (of total dataset).
    num_workers : int
        Number of worker processes for data loading.
    random_state : int
        Random seed for reproducible splits (default: 42).

    Returns
    -------
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    test_loader : DataLoader
        Test data loader.
        
    Notes
    -----
    The same random_state guarantees reproducible splits across different runs.
    Use get_matching_pairs(split, train_loader, val_loader, test_loader) to
    extract only the positive (matching) pairs from any split.
    """
    # First split: train+val vs test
    X_temp, X_test, T_temp, T_test, Y_temp, Y_test = train_test_split(
        features_all, thetas_all, labels_all, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True
    )
    
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)  
    X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(
        X_temp, T_temp, Y_temp, 
        test_size=val_fraction, 
        random_state=random_state, 
        shuffle=True
    )

    # Build datasets
    train_dataset = TensorDataset(X_train, T_train, Y_train)
    val_dataset   = TensorDataset(X_val, T_val, Y_val)
    test_dataset  = TensorDataset(X_test, T_test, Y_test)

    # Build loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_matching_pairs(
    split: str,
    train_loader: Optional[DataLoader],
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract only the matching (positive) pairs (X, T) from a given split.
    
    This function filters out negative (shuffled) pairs created by make_negative_pairs()
    and returns only the true (feature, parameter) pairs. The returned data is 
    deterministic and will be identical across runs if the same dataloaders are used.

    Parameters
    ----------
    split : str
        Must be one of {"train", "val", "test"}.
    train_loader : DataLoader
        Training loader from make_dataloaders().
    val_loader : DataLoader
        Validation loader from make_dataloaders().
    test_loader : DataLoader
        Test loader from make_dataloaders().

    Returns
    -------
    X : torch.Tensor
        Matching features for the specified split.
    T : torch.Tensor
        Corresponding parameter values.
        
    Examples
    --------
    >>> # Get validation matching pairs
    >>> X_val, T_val = get_matching_pairs("val", train_loader, val_loader, test_loader)
    >>> # Get test matching pairs
    >>> X_test, T_test = get_matching_pairs("test", train_loader, val_loader, test_loader)
    """
    if split == "train":
        dataset = train_loader.dataset
    elif split == "val":
        dataset = val_loader.dataset
    elif split == "test":
        dataset = test_loader.dataset
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    X_all, T_all, Y_all = dataset.tensors
    mask_pos = (Y_all == 1)

    X = X_all[mask_pos]
    T = T_all[mask_pos]

    return X, T



# ============================================================================
# DATASET LOADING AND FILTERING
# ============================================================================

def load_matching_pairs(
    halo_choice: str,
    n_train: int,
    datatag: str,
    split: str = "val",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load dataset and extract matching pairs from specified split.
    
    This function ensures reproducible data loading by using fixed random_state
    in make_dataloaders(). The same parameters will always return the same split.
    
    Parameters
    ----------
    halo_choice : str
        Halo model name (e.g., 'default', 'shm', 'shmpp').
    n_train : int
        Number of training samples used.
    datatag : str
        Data category tag (e.g., 'low', 'mid', 'high').
    split : str
        Which split to return: 'train', 'val', or 'test' (default: 'val').
    
    Returns
    -------
    X : torch.Tensor
        Features from the specified split (matching pairs only).
    T : torch.Tensor
        Parameters from the specified split (matching pairs only).
    
    """
    datapath = f"data/datasets/wimpy/{halo_choice}/wimpy_n{n_train}_{datatag}_{halo_choice}.pt"
    try:
        print(f"Loading dataset from:\n{datapath}\n")
        data = torch.load(datapath, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at path: {datapath}")

    theta_tensor = data["theta"]
    feature_tensor = data["features"]

    X_all, T_all, Y_all = make_negative_pairs(feature_tensor, theta_tensor)

    train_loader, val_loader, test_loader = make_dataloaders(X_all, T_all, Y_all)

    X, T = get_matching_pairs(
        split=split,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    return X, T


def filter_by_parameter_ranges(
    T: torch.Tensor,
    X: torch.Tensor,
    logm_window: Optional[Tuple[float, float]] = None,
    logcp_window: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter samples based on parameter ranges in log10 space.
    
    Parameters
    ----------
    T : torch.Tensor
        Parameters of shape (n_samples, 2) where [:, 0] = log10(mchi), [:, 1] = log10(cp).
    X : torch.Tensor
        Features corresponding to T.
    logm_window : Optional[Tuple[float, float]]
        (min, max) range for log10(mass). If None, no filtering on mass.
    logcp_window : Optional[Tuple[float, float]]
        (min, max) range for log10(coupling). If None, no filtering on coupling.
    
    Returns
    -------
    X_filtered : torch.Tensor
        Filtered features.
    T_filtered : torch.Tensor
        Filtered parameters.
    """
    mask = torch.ones(len(T), dtype=torch.bool)

    if logm_window is not None:
        logm_min, logm_max = logm_window
        mask &= (T[:, 0] >= logm_min) & (T[:, 0] <= logm_max)

    if logcp_window is not None:
        logcp_min, logcp_max = logcp_window
        mask &= (T[:, 1] >= logcp_min) & (T[:, 1] <= logcp_max)

    return X[mask], T[mask]



# ============================================================================
# S1S2 HISTOGRAM UTILITIES
# ============================================================================

def clean_nan_events(energies_list, cs1cs2_list, xyz_list=None):
    """
    Remove events containing NaNs from cs1cs2_list and energies_list.
    Optionally filters xyz_list if provided, ensuring all arrays remain aligned.
    
    Args:
        energies_list: Mandatory list of energy arrays.
        cs1cs2_list: Mandatory list of cs1/cs2 arrays.
        xyz_list: Optional list of xyz position arrays (default: None).
    
    Returns:
        If xyz_list is None: (cleaned_energies, cleaned_cs1cs2)
        If xyz_list is provided: (cleaned_energies, cleaned_cs1cs2, cleaned_xyz)
    """
    cleaned_cs1cs2 = []
    cleaned_energies = []
    cleaned_xyz = [] if xyz_list is not None else None

    iterator = zip(energies_list, cs1cs2_list, xyz_list) if xyz_list is not None else zip(energies_list, cs1cs2_list)

    for item in iterator:
        if xyz_list is not None:
            energies, cs1cs2, xyz = item
        else:
            energies, cs1cs2 = item
            xyz = None

        # Case: spectrum is None or empty
        if cs1cs2 is None or len(cs1cs2) == 0:
            cleaned_cs1cs2.append(np.empty((0, 2)))
            cleaned_energies.append(np.empty((0,)))
            if xyz_list is not None:
                cleaned_xyz.append(np.empty((0, 3)))
            continue

        # Create mask: keep rows that are finite in BOTH arrays
        mask = np.isfinite(cs1cs2).all(axis=1)

        # Filter arrays
        cs1cs2_clean = cs1cs2[mask]
        energies_clean = energies[mask]
        xyz_clean = xyz[mask] if xyz is not None else None

        # If empty after cleaning → standard empties
        if len(cs1cs2_clean) == 0:
            cleaned_cs1cs2.append(np.empty((0, 2)))
            cleaned_energies.append(np.empty((0,)))
            if xyz_list is not None:
                cleaned_xyz.append(np.empty((0, 3)))
        else:
            cleaned_cs1cs2.append(cs1cs2_clean)
            cleaned_energies.append(energies_clean)
            if xyz_list is not None:
                cleaned_xyz.append(xyz_clean)

    if xyz_list is not None:
        return cleaned_energies, cleaned_cs1cs2, cleaned_xyz
    else:
        return cleaned_energies, cleaned_cs1cs2


class SignalPlusBackgroundDataset(torch.utils.data.Dataset):

    """
    Dataset that generates 2D histograms on-the-fly from signal + background events.

    When used with DataLoader:
    - DataLoader iterates through indices and calls __getitem__ for batches
    - Each epoch gets different background realizations (if reproducible=False)
    - For validation (reproducible=True), same signal → same background always
    """
    
    def __init__(
        self,
        signal_events,        # list of np.ndarray or torch.Tensor
        theta,                # np.ndarray or torch.Tensor [N_samples, dim_theta]
        background_events,    # np.ndarray or torch.Tensor [N_bg_total, 2]
        mu_bg=0,
        s1_bins=30,
        s2_bins=30,
        s1_range=(0, 100),
        s2_range=(10**2.1, 10**4.1),
        dtype=torch.float32,
        reproducible=False,
        rng_seed=42
    ):
        
        # Convert to tensors
        self.signal_events = [
            torch.as_tensor(s, dtype=dtype) for s in signal_events
        ]
        self.theta = torch.as_tensor(theta, dtype=dtype)
        self.background_events = torch.as_tensor(background_events, dtype=dtype)

        self.mu_bg = mu_bg
        self.s1_bins = s1_bins
        self.s2_bins = s2_bins
        self.s1_range = s1_range
        self.s2_range = s2_range
        self.dtype = dtype

        # Precompute bin edges
        self.s1_edges = np.linspace(*s1_range, s1_bins + 1)
        self.s2_edges = np.logspace(np.log10(s2_range[0]), np.log10(s2_range[1]), s2_bins + 1)

        self.n_bg_total = self.background_events.shape[0]

        # Optional reproducible background sampling
        self.reproducible = reproducible
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed) if reproducible else np.random.default_rng()

    def sample_background(self):
        """Sample background events from the pool."""
        N_bg = self.rng.poisson(self.mu_bg)
        if N_bg == 0:
            return None
        idx = self.rng.integers(0, self.n_bg_total, size=N_bg)
        return self.background_events[idx]

    def histogram(self, events):
        """Convert events to 2D histogram."""
        if isinstance(events, torch.Tensor):
            events_np = events.cpu().numpy()
        else:
            events_np = events

        cS1, cS2 = events_np[:, 0], events_np[:, 1]
        H, _, _ = np.histogram2d(cS1, cS2, bins=[self.s1_edges, self.s2_edges])
        
        return torch.tensor(H, dtype=self.dtype)

    def __getitem__(self, idx):
        sig = self.signal_events[idx]
        
        # Reseed per-index for reproducible validation
        if self.reproducible:
            self.rng = np.random.default_rng(self.rng_seed + idx)
        
        bg = self.sample_background()
        if bg is not None:
            events = torch.cat([sig, bg], dim=0)
        else:
            events = sig

        x = self.histogram(events)
        y = self.theta[idx]

        return x, y

    def __len__(self):
        return len(self.signal_events)



# ============================================================================
# S1S2 VALIDATION DATALOADER
# ============================================================================

def load_s1s2_valloader(
    signal_pt: str,
    bg_csv: str,
    mu_bg: int = 0,
    bins: int = 30,
    n_eval: int = 5000,
    batch_size: int = 128,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load S1S2 signal data, split off a validation set, add background,
    and return a DataLoader of positive/negative pairs for evaluation.

    Parameters
    ----------
    signal_pt : str
        Path to the .pt file containing signal data (keys: "events", "cs1cs2", "theta").
    bg_csv : str
        Path to the CSV file with background events (columns: cS1, cS2).
    mu_bg : int
        Mean number of background events to add per spectrum (0 = signal-only).
    bins : int
        Number of bins per axis for histogramming.
    n_eval : int
        Maximum number of spectra to use for the validation loader.
    batch_size : int
        Batch size for the returned DataLoader.
    test_size : float
        Fraction of samples held out as validation set.
    random_state : int
        Random seed for reproducibility of the train/test split.

    Returns
    -------
    val_loader : DataLoader
        DataLoader with (features, theta, label) triplets.
    bg_events : np.ndarray
        Background event pool loaded from *bg_csv*.
    """
    

    # Load signal data
    signal_data = torch.load(signal_pt, weights_only=False)
    energies_list = signal_data["events"]
    cs1cs2_list = signal_data["cs1cs2"]
    theta_list = signal_data["theta"]

    # Clean NaNs
    energies_list, cs1cs2_list = clean_nan_events(energies_list, cs1cs2_list)

    # Split into train / validation
    _, cs1cs2_val, _, theta_val = train_test_split(
        cs1cs2_list, theta_list, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Background pool
    bg_events = np.loadtxt(bg_csv, delimiter=",", skiprows=1, usecols=(0, 1))

    # Build validation dataset
    val_dataset = SignalPlusBackgroundDataset(
        signal_events=cs1cs2_val,
        theta=theta_val,
        background_events=bg_events,
        mu_bg=mu_bg,
        s1_bins=bins,
        s2_bins=bins,
        reproducible=True,
        rng_seed=random_state,
    )

    # Subsample to keep evaluation light
    n_eval_eff = min(n_eval, len(val_dataset))
    X_list, T_list = [], []
    for i in range(n_eval_eff):
        x, t = val_dataset[int(i)]
        X_list.append(x.view(-1))
        T_list.append(t)

    feature_tensor = torch.stack(X_list)
    theta_tensor = torch.stack(T_list)

    # Create positive/negative pairs
    X_all, T_all, Y_all = make_negative_pairs(feature_tensor, theta_tensor)

    # Create dataloader for evaluation
    tensor_dataset = TensorDataset(X_all, T_all, Y_all)
    val_loader = DataLoader(tensor_dataset, batch_size=batch_size, num_workers=4)

    print(f"Validation pairs: {len(val_loader.dataset):,}")

    return val_loader, bg_events
