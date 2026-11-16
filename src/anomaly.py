"""Milestone 8 (bonus): autoencoder anomaly detection.

Train a convolutional autoencoder on NORMAL training ECGs only, then use the
reconstruction error (MSE) of an ECG as its anomaly score: high error = unusual.

Run with:  python -m src.anomaly --config configs/autoencoder.yaml
Add --no-test for experiments (reports validation only, test fold untouched).
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import CLASSES, FIG_DIR, LEADS, RESULTS_DIR
from src.dataset import ECGDataset, load_splits
from src.models.autoencoder import ConvAutoencoder
from src.utils import count_parameters, get_device, set_seed


def is_normal(y):
    """Normal = NORM is the ONLY label. Any ECG with MI/STTC/CD/HYP counts as abnormal."""
    return (y[:, 0] == 1) & (y[:, 1:].sum(axis=1) == 0)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """One pass over the normal training ECGs. The target is the input itself."""
    model.train()
    total_loss = 0.0
    for x, _ in loader:  # labels are not used: this is unsupervised
        x = x.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(x), x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def anomaly_scores(model, loader, device):
    """Reconstruction MSE of every ECG (averaged over all leads and time steps)."""
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device)
        error = (model(x) - x) ** 2
        scores.append(error.mean(dim=(1, 2)).cpu())
    return torch.cat(scores).numpy()


def detection_metrics(scores, y):
    """AUROC for 'normal vs. abnormal', overall and against each abnormal class."""
    normal = is_normal(y)
    metrics = {"auroc": float(roc_auc_score(~normal, scores)), "auroc_vs_class": {}}
    for c in range(1, len(CLASSES)):
        # normal ECGs against the ECGs that carry this one abnormal label
        keep = normal | (y[:, c] == 1)
        metrics["auroc_vs_class"][CLASSES[c]] = float(roc_auc_score(~normal[keep], scores[keep]))
    return metrics


def plot_training_curve(history, run_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train loss (normal ECGs)")
    ax.plot(history["epoch"], history["val_loss"], label="val loss (normal ECGs)")
    best = history.loc[history["val_loss"].idxmin()]
    ax.axvline(best["epoch"], color="gray", linestyle="--", label=f"best epoch {int(best['epoch'])}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(f"{run_name}: reconstruction loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"training_curves_{run_name}.png"), dpi=150)
    plt.close(fig)


def plot_score_histogram(scores, y, auroc):
    """Histogram of anomaly scores on the test fold: normal vs. abnormal ECGs."""
    normal = is_normal(y)
    # the scores are skewed (a few very large ones), so use bins that are even on a log scale
    bins = np.logspace(np.log10(scores.min()), np.log10(scores.max()), 50)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores[normal], bins=bins, alpha=0.6, color="tab:blue",
            label=f"normal (n={normal.sum()}, median {np.median(scores[normal]):.3f})")
    ax.hist(scores[~normal], bins=bins, alpha=0.6, color="tab:red",
            label=f"abnormal (n={(~normal).sum()}, median {np.median(scores[~normal]):.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Anomaly score = reconstruction MSE (log scale)")
    ax.set_ylabel("Number of test ECGs")
    ax.set_title(f"Autoencoder anomaly scores on the test fold (AUROC {auroc:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "anomaly_score_histogram.png"), dpi=150)
    plt.close(fig)


def pick_examples(scores, y):
    """A typical normal ECG (median score of the normals) and a badly rebuilt
    abnormal ECG (95th percentile of the abnormal scores). Returns two indices."""
    normal = is_normal(y)
    normal_idx, abnormal_idx = np.where(normal)[0], np.where(~normal)[0]
    good = normal_idx[np.argsort(scores[normal_idx])[len(normal_idx) // 2]]
    poor = abnormal_idx[np.argsort(scores[abnormal_idx])[int(0.95 * len(abnormal_idx))]]
    return int(good), int(poor)


@torch.no_grad()
def plot_reconstructions(model, split, scores, device, leads=("II", "V2")):
    """Original vs. reconstructed signal for one good and one poor reconstruction."""
    model.eval()
    good, poor = pick_examples(scores, split["y"])
    time_axis = np.arange(1000) / 100  # seconds
    fig, axes = plt.subplots(len(leads), 2, figsize=(14, 5.5), sharex=True)
    for col, (i, kind) in enumerate([(good, "Good reconstruction"), (poor, "Poor reconstruction")]):
        x = split["x"][i]  # (1000, 12), standardized
        x_tensor = torch.from_numpy(x.T.copy()).unsqueeze(0).to(device)  # (1, 12, 1000)
        rebuilt = model(x_tensor)[0].cpu().numpy().T
        labels = "+".join(c for c, flag in zip(CLASSES, split["y"][i]) if flag == 1)
        for row, lead in enumerate(leads):
            ax = axes[row, col]
            k = LEADS.index(lead)
            ax.plot(time_axis, x[:, k], color="black", linewidth=0.8, label="original")
            ax.plot(time_axis, rebuilt[:, k], color="tab:red", linewidth=0.8, label="reconstruction")
            ax.set_ylabel(f"Lead {lead}\n(standardized)")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"{kind}: ECG {split['ecg_ids'][i]} ({labels}), "
                             f"score {scores[i]:.3f}")
                ax.legend(loc="upper right", fontsize=8)
        axes[-1, col].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "anomaly_reconstruction_examples.png"), dpi=150)
    plt.close(fig)
    return good, poor


def supervised_reference(y_test):
    """For comparison: how well does the supervised CNN separate normal from abnormal?
    Its score is 1 - P(NORM). Uses the test predictions saved in Milestone 3."""
    path = os.path.join(RESULTS_DIR, "cnn", "test_probs.npy")
    if not os.path.exists(path):
        return None
    probs = np.load(path)
    return float(roc_auc_score(~is_normal(y_test), 1 - probs[:, 0]))


def energy_baseline(x, y):
    """Sanity check: a 'detector' with no model at all. Its score is just the signal
    energy (mean squared amplitude). Big signals are harder to rebuild, so if the
    autoencoder were no better than this, it would only be measuring amplitude."""
    energy = (x ** 2).mean(axis=(1, 2))
    return float(roc_auc_score(~is_normal(y), energy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-test", action="store_true",
                        help="report validation only (for experiments; keeps the test fold unused)")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    use_amp = device.type == "cuda"  # mixed precision only on CUDA
    run_name = config["run_name"]
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    # Same preprocessed data as the classifiers (filtered, standardized with train stats).
    splits = load_splits()
    train_normal = is_normal(splits["train"]["y"])
    val_normal = is_normal(splits["val"]["y"])
    print(f"Normal ECGs: train {train_normal.sum()} of {len(train_normal)}, "
          f"val {val_normal.sum()} of {len(val_normal)}")

    generator = torch.Generator().manual_seed(config["seed"])  # reproducible shuffling
    loaders = {
        # the autoencoder only ever trains on NORMAL training ECGs
        "train_normal": DataLoader(
            ECGDataset(splits["train"]["x"][train_normal], splits["train"]["y"][train_normal]),
            batch_size=config["batch_size"], shuffle=True, generator=generator),
        # early stopping also looks at normal ECGs only, so no abnormal label is used in training
        "val_normal": DataLoader(
            ECGDataset(splits["val"]["x"][val_normal], splits["val"]["y"][val_normal]),
            batch_size=256, shuffle=False),
        "val": DataLoader(ECGDataset(splits["val"]["x"], splits["val"]["y"]),
                          batch_size=256, shuffle=False),
        "test": DataLoader(ECGDataset(splits["test"]["x"], splits["test"]["y"]),
                           batch_size=256, shuffle=False),
    }

    model = ConvAutoencoder(**config.get("model_args", {})).to(device)
    n_params = count_parameters(model)
    print(f"Run '{run_name}' on {device}: {n_params:,} parameters")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history = []
    best_loss, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, loaders["train_normal"], criterion, optimizer,
                                     scaler, device, use_amp)
        # mean score of the normal validation ECGs = validation reconstruction loss
        val_loss = float(anomaly_scores(model, loaders["val_normal"], device).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "time_s": time.time() - epoch_start})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"({time.time() - epoch_start:.1f}s)")

        # Early stopping on the validation reconstruction loss (lower is better).
        if val_loss < best_loss:
            best_loss, best_epoch, epochs_without_improvement = val_loss, epoch, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"Early stopping: no improvement for {config['patience']} epochs.")
                break

    train_time = time.time() - start
    history = pd.DataFrame(history)
    history.to_csv(os.path.join(run_dir, "history.csv"), index=False)
    plot_training_curve(history, run_name)
    print(f"Best epoch {best_epoch} (val loss {best_loss:.4f}), training time {train_time:.0f}s")

    # ---- Evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    metrics = {"params": n_params, "train_time_s": round(train_time, 1), "best_epoch": best_epoch,
               "epochs_run": len(history), "n_train_normal": int(train_normal.sum()),
               "val_normal_loss": best_loss}

    val_scores = anomaly_scores(model, loaders["val"], device)
    metrics["val"] = detection_metrics(val_scores, splits["val"]["y"])
    print(f"Validation AUROC (normal vs. abnormal): {metrics['val']['auroc']:.4f}")

    if not args.no_test:
        y_test = splits["test"]["y"]
        test_scores = anomaly_scores(model, loaders["test"], device)
        np.save(os.path.join(run_dir, "test_scores.npy"), test_scores)
        metrics["test"] = detection_metrics(test_scores, y_test)
        test_normal = is_normal(y_test)
        metrics["test"]["n_normal"] = int(test_normal.sum())
        metrics["test"]["n_abnormal"] = int((~test_normal).sum())
        metrics["test"]["median_score_normal"] = float(np.median(test_scores[test_normal]))
        metrics["test"]["median_score_abnormal"] = float(np.median(test_scores[~test_normal]))
        metrics["test"]["supervised_cnn_auroc"] = supervised_reference(y_test)
        metrics["test"]["signal_energy_auroc"] = energy_baseline(splits["test"]["x"], y_test)
        # the single highest score: worth looking at, it is often a noisy recording
        worst = int(np.argmax(test_scores))
        metrics["test"]["highest_score_ecg_id"] = int(splits["test"]["ecg_ids"][worst])
        metrics["test"]["highest_score_is_normal"] = bool(test_normal[worst])
        print(f"Test AUROC (normal vs. abnormal): {metrics['test']['auroc']:.4f}")
        for name, value in metrics["test"]["auroc_vs_class"].items():
            print(f"  normal vs. {name}: {value:.4f}")
        print(f"  for comparison, signal energy alone (no model): "
              f"{metrics['test']['signal_energy_auroc']:.4f}")
        if metrics["test"]["supervised_cnn_auroc"] is not None:
            print(f"  for comparison, supervised CNN with 1 - P(NORM): "
                  f"{metrics['test']['supervised_cnn_auroc']:.4f}")

        plot_score_histogram(test_scores, y_test, metrics["test"]["auroc"])
        good, poor = plot_reconstructions(model, splits["test"], test_scores, device)
        metrics["test"]["example_good_ecg_id"] = int(splits["test"]["ecg_ids"][good])
        metrics["test"]["example_poor_ecg_id"] = int(splits["test"]["ecg_ids"][poor])

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {os.path.join(run_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
