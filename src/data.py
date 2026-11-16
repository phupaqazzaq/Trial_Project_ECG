"""Milestone 1: load PTB-XL, build the 5-class labels, split by fold, cache to disk.

Run with:  python -m src.data
"""

import ast
import os

import matplotlib

matplotlib.use("Agg")  # draw figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

DATA_DIR = "data/ptb-xl"
CACHE_DIR = "data/cache"
FIG_DIR = "figures"
RESULTS_DIR = "results"

# Fixed label order used everywhere in the project.
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Order of the 12 leads inside every PTB-XL record.
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Official split: folds 1-8 train, fold 9 validation, fold 10 test.
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLD = 9
TEST_FOLD = 10


def check_dataset():
    """Stop early with a clear message if the dataset has not been downloaded."""
    needed = ["ptbxl_database.csv", "scp_statements.csv", "records100"]
    missing = [n for n in needed if not os.path.exists(os.path.join(DATA_DIR, n))]
    if missing:
        raise FileNotFoundError(
            f"PTB-XL not found in {DATA_DIR} (missing: {missing}). "
            "Download it from https://physionet.org/content/ptb-xl/1.0.3/ "
            "and unzip into data/ptb-xl/ (see README.md)."
        )


def load_metadata():
    """Read the two CSV files and attach the list of superclasses to every ECG."""
    df = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"), index_col="ecg_id")
    # scp_codes is stored as text like "{'NORM': 100.0, 'SR': 0.0}" -> turn into a dict
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # scp_statements.csv says which codes are diagnostic and which superclass they belong to
    statements = pd.read_csv(os.path.join(DATA_DIR, "scp_statements.csv"), index_col=0)
    statements = statements[statements["diagnostic"] == 1]

    def codes_to_superclasses(codes):
        found = set()
        for code in codes:
            if code in statements.index:
                found.add(statements.loc[code, "diagnostic_class"])
        return sorted(found)

    df["superclasses"] = df["scp_codes"].apply(codes_to_superclasses)

    # Drop ECGs that have no diagnostic superclass at all.
    n_before = len(df)
    df = df[df["superclasses"].apply(len) > 0]
    print(f"ECGs in database: {n_before}, kept with >=1 superclass: {len(df)}")
    return df


def make_multi_hot(df):
    """Turn the superclass lists into a (N, 5) array of 0/1 in the order of CLASSES."""
    y = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for row, labels in enumerate(df["superclasses"]):
        for label in labels:
            y[row, CLASSES.index(label)] = 1.0
    return y


def load_signals(df):
    """Load the 100 Hz waveform of every ECG -> array (N, 1000, 12)."""
    signals = np.zeros((len(df), 1000, 12), dtype=np.float32)
    for row, filename in enumerate(tqdm(df["filename_lr"], desc="Loading ECGs")):
        signal, _ = wfdb.rdsamp(os.path.join(DATA_DIR, filename))
        signals[row] = signal
    return signals


def build_cache():
    """Load everything once and save it as .npy files so later runs are fast."""
    check_dataset()
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = load_metadata()
    y = make_multi_hot(df)
    x = load_signals(df)

    np.save(os.path.join(CACHE_DIR, "X.npy"), x)
    np.save(os.path.join(CACHE_DIR, "Y.npy"), y)
    np.save(os.path.join(CACHE_DIR, "folds.npy"), df["strat_fold"].to_numpy())
    np.save(os.path.join(CACHE_DIR, "ecg_ids.npy"), df.index.to_numpy())
    # Age and sex are needed later for the LLM report (sex: 0 = male, 1 = female).
    df[["age", "sex", "strat_fold"]].to_csv(os.path.join(CACHE_DIR, "meta.csv"))
    print(f"Saved cache to {CACHE_DIR}: X {x.shape}, Y {y.shape}")


def load_cache(signal_file="X.npy"):
    """Return the cached arrays. Builds the cache first if it does not exist."""
    if not os.path.exists(os.path.join(CACHE_DIR, "X.npy")):
        build_cache()
    return {
        "X": np.load(os.path.join(CACHE_DIR, signal_file)),
        "Y": np.load(os.path.join(CACHE_DIR, "Y.npy")),
        "folds": np.load(os.path.join(CACHE_DIR, "folds.npy")),
        "ecg_ids": np.load(os.path.join(CACHE_DIR, "ecg_ids.npy")),
    }


def split_masks(folds):
    """Boolean masks for train / val / test based on the official strat_fold column."""
    return {
        "train": np.isin(folds, TRAIN_FOLDS),
        "val": folds == VAL_FOLD,
        "test": folds == TEST_FOLD,
    }


# ---------------------------------------------------------------- exploration


def print_summary(y, masks):
    """Print split sizes, label counts and the share of multi-label ECGs."""
    rows = []
    for split, mask in masks.items():
        counts = y[mask].sum(axis=0).astype(int)
        rows.append({"split": split, "n_ecgs": int(mask.sum()), **dict(zip(CLASSES, counts))})
    table = pd.DataFrame(rows).set_index("split")
    print("\nECGs and label counts per split:")
    print(table.to_string())

    n_labels = y.sum(axis=1)
    print(f"\nMulti-label ECGs (2 or more superclasses): {100 * (n_labels >= 2).mean():.1f}%")
    for k in range(1, 5):
        print(f"  ECGs with exactly {k} label(s): {int((n_labels == k).sum())}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "class_counts.csv"))
    return table


def plot_class_distribution(table):
    """Bar chart: number of ECGs per superclass in train / val / test."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.27
    positions = np.arange(len(CLASSES))
    for i, split in enumerate(["train", "val", "test"]):
        counts = table.loc[split, CLASSES].to_numpy()
        bars = ax.bar(positions + (i - 1) * width, counts, width, label=split)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Number of ECGs")
    ax.set_title("PTB-XL superclass counts per split (an ECG can have several labels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)


def plot_example_ecgs(x, y, ecg_ids):
    """One example ECG per class (an ECG with only that single label), leads II and V2."""
    time = np.arange(1000) / 100  # seconds
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(13, 10), sharex=True)
    for row, name in enumerate(CLASSES):
        only_this_class = (y[:, row] == 1) & (y.sum(axis=1) == 1)
        idx = np.where(only_this_class)[0][0]
        for col, lead in enumerate(["II", "V2"]):
            ax = axes[row, col]
            ax.plot(time, x[idx, :, LEADS.index(lead)], linewidth=0.7, color="black")
            ax.set_title(f"{name} - lead {lead} (ecg_id {ecg_ids[idx]})", fontsize=9)
            ax.set_ylabel("mV")
            ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "example_ecgs.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_cache()
    x, y = data["X"], data["Y"]
    print(f"Signals: {x.shape} {x.dtype}, labels: {y.shape} {y.dtype}")

    masks = split_masks(data["folds"])
    table = print_summary(y, masks)
    plot_class_distribution(table)
    plot_example_ecgs(x, y, data["ecg_ids"])
    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
