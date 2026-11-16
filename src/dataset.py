"""Milestone 2: PyTorch Dataset and DataLoaders.

Run with:  python -m src.dataset   (prints the shape of one batch as a check)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import load_cache, split_masks
from src.preprocess import FILTERED_FILE, STATS_FILE, standardize
from src.utils import SEED, set_seed


class ECGDataset(Dataset):
    """Holds already-preprocessed ECGs in memory.

    Each item is (signal, label):
      signal: float tensor (12, 1000)  -> (leads, time), the layout Conv1d expects
      label:  float tensor (5,)        -> multi-hot [NORM, MI, STTC, CD, HYP]
    """

    def __init__(self, x, y):
        # stored as (N, 1000, 12); swap the last two axes to get (N, 12, 1000)
        self.x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1)))
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_splits():
    """Load filtered signals, standardize with TRAIN statistics, return arrays per split."""
    data = load_cache(signal_file=FILTERED_FILE)
    stats = np.load(STATS_FILE)  # mean/std were computed on the training folds only
    masks = split_masks(data["folds"])

    splits = {}
    for name, mask in masks.items():
        splits[name] = {
            "x": standardize(data["X"][mask], stats["mean"], stats["std"]),
            "y": data["Y"][mask],
            "ecg_ids": data["ecg_ids"][mask],
        }
    return splits


def get_dataloaders(batch_size=64, seed=SEED):
    """Return {'train': ..., 'val': ..., 'test': ...} DataLoaders plus the raw splits."""
    splits = load_splits()
    generator = torch.Generator().manual_seed(seed)  # makes the shuffling reproducible
    loaders = {
        "train": DataLoader(ECGDataset(splits["train"]["x"], splits["train"]["y"]),
                            batch_size=batch_size, shuffle=True, generator=generator),
        "val": DataLoader(ECGDataset(splits["val"]["x"], splits["val"]["y"]),
                          batch_size=256, shuffle=False),
        "test": DataLoader(ECGDataset(splits["test"]["x"], splits["test"]["y"]),
                           batch_size=256, shuffle=False),
    }
    return loaders, splits


def main():
    set_seed()
    loaders, splits = get_dataloaders()
    for name, loader in loaders.items():
        print(f"{name}: {len(loader.dataset)} ECGs, {len(loader)} batches")
    x, y = next(iter(loaders["train"]))
    print(f"One training batch: x {tuple(x.shape)} {x.dtype}, y {tuple(y.shape)} {y.dtype}")

    # Check: after standardization the TRAIN data has mean ~0 and std ~1 per lead.
    # Val/test are close to that but not exact, because they use the train statistics.
    for name in ["train", "val", "test"]:
        xs = splits[name]["x"]
        print(f"{name}: per-lead mean in [{xs.mean(axis=(0, 1)).min():+.3f}, {xs.mean(axis=(0, 1)).max():+.3f}], "
              f"std in [{xs.std(axis=(0, 1)).min():.3f}, {xs.std(axis=(0, 1)).max():.3f}]")


if __name__ == "__main__":
    main()
