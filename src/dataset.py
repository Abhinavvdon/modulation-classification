"""Dataset and DataLoader utilities for Modulation Classification.

Provides:
- ModulationDataset (PyTorch Dataset for I/Q signal frames)
- Stratified train/val/test splitting
- DataLoader creation
- Serialization / Deserialization to NumPy .npz archive format
"""

import os
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


class ModulationDataset(Dataset):
    """PyTorch Dataset wrapper for raw baseband I/Q signal arrays."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        snrs: np.ndarray
    ):
        """Args:

        signals: Array of shape (N, 2, sequence_length)
        labels: 1D array of class indices (N,)
        snrs: 1D array of SNR values in dB (N,)
        """
        assert len(signals) == len(labels) == len(snrs), "Dimension mismatch between inputs"
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()
        self.snrs = torch.from_numpy(snrs).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx], self.snrs[idx]


def split_dataset(
    signals: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Performs stratified train/val/test split across (modulation, SNR) pairs."""
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"

    # Create composite stratification key combining label and SNR
    composite_keys = [f"{lbl}_{snr}" for lbl, snr in zip(labels, snrs)]
    unique_keys, key_indices = np.unique(composite_keys, return_inverse=True)

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_seed)
    train_idx, temp_idx = next(sss1.split(signals, key_indices))

    # Second split: Val vs Test
    relative_test_ratio = test_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=random_seed)
    val_sub_idx, test_sub_idx = next(sss2.split(signals[temp_idx], key_indices[temp_idx]))
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    return {
        "train": (signals[train_idx], labels[train_idx], snrs[train_idx]),
        "val": (signals[val_idx], labels[val_idx], snrs[val_idx]),
        "test": (signals[test_idx], labels[test_idx], snrs[test_idx]),
    }


def create_dataloaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int = 64,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """Wraps train/val/test splits in PyTorch DataLoaders."""
    train_ds = ModulationDataset(*splits["train"])
    val_ds = ModulationDataset(*splits["val"])
    test_ds = ModulationDataset(*splits["test"])

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def save_dataset_npz(
    filepath: str,
    signals: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    class_names: List[str]
) -> None:
    """Saves raw dataset arrays and class metadata into a compressed .npz archive."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(
        filepath,
        signals=signals,
        labels=labels,
        snrs=snrs,
        class_names=np.array(class_names)
    )


def load_dataset_npz(
    filepath: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads dataset from compressed .npz archive."""
    data = np.load(filepath)
    signals = data["signals"]
    labels = data["labels"]
    snrs = data["snrs"]
    class_names = [str(c) for c in data["class_names"]]
    return signals, labels, snrs, class_names
