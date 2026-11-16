"""Milestone 3: training loop with early stopping, checkpointing and logging.

Run with:  python -m src.train --config configs/cnn.yaml
Add --no-test for quick experiments (reports validation only, test fold untouched).
"""

import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data import FIG_DIR, RESULTS_DIR
from src.dataset import get_dataloaders
from src.evaluate import auroc_scores, evaluate_run, predict
from src.models import build_model
from src.utils import count_parameters, get_device, set_seed


def lr_factor(epoch, warmup_epochs, total_epochs, cosine_decay):
    """Multiplier for the learning rate at a given epoch (0-based).

    Warmup: rise linearly from a small value to 1 over the first `warmup_epochs`.
    Cosine decay: afterwards fall smoothly towards 0 following half a cosine wave.
    """
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    if not cosine_decay:
        return 1.0
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, grad_clip):
    """One pass over the training data. Returns the average training loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # autocast = mixed precision: faster on CUDA GPUs, switched off elsewhere
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)  # clip the real (unscaled) gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validation_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += criterion(model(x), y).item() * len(x)
    return total / len(loader.dataset)


def plot_training_curves(history, run_name):
    """Loss curves (train vs. val) and validation macro AUROC per epoch."""
    best = history.loc[history["val_macro_auroc"].idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title(f"{run_name}: loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_macro_auroc"], color="tab:green")
    axes[1].axvline(best["epoch"], color="gray", linestyle="--",
                    label=f"best epoch {int(best['epoch'])} ({best['val_macro_auroc']:.4f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro AUROC")
    axes[1].set_title(f"{run_name}: validation macro AUROC")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"training_curves_{run_name}.png"), dpi=150)
    plt.close(fig)


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

    loaders, splits = get_dataloaders(config["batch_size"], config["seed"])
    model = build_model(config).to(device)
    n_params = count_parameters(model)
    print(f"Run '{run_name}' on {device}: {n_params:,} parameters")

    # Loss: one sigmoid + binary cross-entropy per class (multi-label problem).
    # Optional pos_weight = (#negative / #positive) per class, computed on TRAIN labels,
    # which makes mistakes on rare classes count more.
    pos_weight = None
    if config.get("use_pos_weight", False):
        y_train = torch.from_numpy(splits["train"]["y"])
        pos_weight = ((1 - y_train).sum(0) / y_train.sum(0)).to(device)
        print("pos_weight:", [round(v, 2) for v in pos_weight.tolist()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: lr_factor(epoch, config.get("warmup_epochs", 0), config["epochs"],
                                config.get("cosine_decay", False)),
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history = []
    best_auroc, best_epoch, epochs_without_improvement = 0.0, 0, 0
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler,
                                     device, use_amp, config.get("grad_clip", 0))
        val_loss = validation_loss(model, loaders["val"], criterion, device)
        val_probs, val_labels = predict(model, loaders["val"], device)
        val_auroc, _ = auroc_scores(val_labels, val_probs)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_auroc": val_auroc, "lr": lr_now,
                        "time_s": time.time() - epoch_start})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val macro AUROC {val_auroc:.4f}  lr {lr_now:.2e}  ({time.time() - epoch_start:.1f}s)")

        # Early stopping: keep the checkpoint with the best validation macro AUROC,
        # stop when it has not improved for `patience` epochs.
        if val_auroc > best_auroc:
            best_auroc, best_epoch, epochs_without_improvement = val_auroc, epoch, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"Early stopping: no improvement for {config['patience']} epochs.")
                break

    train_time = time.time() - start
    history = pd.DataFrame(history)
    history.to_csv(os.path.join(run_dir, "history.csv"), index=False)
    plot_training_curves(history, run_name)
    with open(os.path.join(run_dir, "train_info.json"), "w") as f:
        json.dump({"params": n_params, "train_time_s": round(train_time, 1),
                   "best_epoch": best_epoch, "epochs_run": len(history),
                   "device": str(device)}, f, indent=2)
    print(f"Best epoch {best_epoch} (val macro AUROC {best_auroc:.4f}), "
          f"total training time {train_time:.0f}s")

    evaluate_run(run_name, use_test=not args.no_test)


if __name__ == "__main__":
    main()
