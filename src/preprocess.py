"""Milestone 2: bandpass filtering + per-lead standardization.

Run with:  python -m src.preprocess
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

from src.data import CACHE_DIR, FIG_DIR, LEADS, load_cache, split_masks

FS = 100  # sampling rate in Hz
FILTERED_FILE = "X_filtered.npy"
STATS_FILE = os.path.join(CACHE_DIR, "norm_stats.npz")


def bandpass_filter(x, low=0.5, high=40.0, fs=FS, order=3):
    """Keep only 0.5-40 Hz. x has shape (..., time, leads); we filter along time.

    - below 0.5 Hz: slow baseline drift (breathing, electrode movement)
    - above 40 Hz: muscle noise and power-line interference (50 Hz)
    filtfilt runs the filter forwards and backwards, so peaks are not shifted in time.
    """
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    return filtfilt(b, a, x, axis=-2).astype(np.float32)


def filter_all(x, chunk=2000):
    """Filter the whole dataset in chunks to keep memory use small."""
    out = np.zeros_like(x)
    for start in range(0, len(x), chunk):
        out[start:start + chunk] = bandpass_filter(x[start:start + chunk])
    return out


def compute_norm_stats(x_train):
    """Mean and std of each lead, computed over all training ECGs and time steps."""
    mean = x_train.mean(axis=(0, 1))  # shape (12,)
    std = x_train.std(axis=(0, 1))    # shape (12,)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x, mean, std):
    """(x - mean) / std for each lead. Broadcasting matches the last axis (leads)."""
    return (x - mean) / std


def plot_before_after(raw, filtered, ecg_id, lead="II"):
    """Show one lead before and after filtering."""
    time = np.arange(raw.shape[0]) / FS
    lead_idx = LEADS.index(lead)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    axes[0].plot(time, raw[:, lead_idx], color="tab:red", linewidth=0.8)
    axes[0].set_title(f"Raw signal - lead {lead} (ecg_id {ecg_id})")
    axes[1].plot(time, filtered[:, lead_idx], color="tab:blue", linewidth=0.8)
    axes[1].set_title("After 0.5-40 Hz bandpass filter (baseline drift and noise removed)")
    for ax in axes:
        ax.set_ylabel("mV")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "preprocessing_before_after.png"), dpi=150)
    plt.close(fig)


def main():
    data = load_cache()
    x_raw = data["X"]
    masks = split_masks(data["folds"])

    print("Filtering all ECGs (0.5-40 Hz Butterworth, order 3)...")
    x_filtered = filter_all(x_raw)
    np.save(os.path.join(CACHE_DIR, FILTERED_FILE), x_filtered)

    # IMPORTANT: statistics come from the training folds only (no leakage from val/test).
    mean, std = compute_norm_stats(x_filtered[masks["train"]])
    np.savez(STATS_FILE, mean=mean, std=std)
    print("Per-lead mean (train only):", np.round(mean, 4))
    print("Per-lead std  (train only):", np.round(std, 4))

    # For the figure, pick the training ECG whose lead II drifts the most,
    # so the effect of the filter is easy to see.
    lead_ii = x_raw[masks["train"]][:, :, LEADS.index("II")]
    drift = np.abs(lead_ii[:, :100].mean(axis=1) - lead_ii[:, -100:].mean(axis=1))
    idx = np.where(masks["train"])[0][np.argsort(drift)[-20]]  # strong drift, not the extreme outlier
    plot_before_after(x_raw[idx], x_filtered[idx], data["ecg_ids"][idx])
    print(f"Saved {FILTERED_FILE}, norm_stats.npz and figures/preprocessing_before_after.png")


if __name__ == "__main__":
    main()
