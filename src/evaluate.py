"""Milestone 3: metrics, threshold tuning and result tables.

Run with:  python -m src.evaluate --run cnn
(train.py calls this automatically at the end of a training run)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.data import CLASSES, RESULTS_DIR
from src.dataset import get_dataloaders
from src.models import build_model
from src.utils import get_device

SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.csv")


@torch.no_grad()
def predict(model, loader, device):
    """Run the model on a whole split. Returns probabilities and true labels, both (N, 5)."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_probs.append(torch.sigmoid(logits).float().cpu().numpy())  # sigmoid: each class independent
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def auroc_scores(labels, probs):
    """AUROC for each class and their plain average (macro AUROC)."""
    per_class = roc_auc_score(labels, probs, average=None)
    return float(per_class.mean()), {c: float(v) for c, v in zip(CLASSES, per_class)}


def tune_thresholds(val_labels, val_probs):
    """For each class, find the probability cut-off that gives the best F1 on VALIDATION."""
    grid = np.arange(0.05, 0.96, 0.01)
    thresholds = {}
    for i, name in enumerate(CLASSES):
        f1s = [f1_score(val_labels[:, i], val_probs[:, i] >= t, zero_division=0) for t in grid]
        thresholds[name] = float(grid[int(np.argmax(f1s))])
    return thresholds


def classification_metrics(labels, probs, thresholds):
    """Per-class F1 / precision / recall after applying the tuned thresholds."""
    out = {}
    for i, name in enumerate(CLASSES):
        pred = probs[:, i] >= thresholds[name]
        out[name] = {
            "f1": float(f1_score(labels[:, i], pred, zero_division=0)),
            "precision": float(precision_score(labels[:, i], pred, zero_division=0)),
            "recall": float(recall_score(labels[:, i], pred, zero_division=0)),
        }
    return out


def update_summary(row):
    """Add (or replace) one row per run in results/summary.csv."""
    new = pd.DataFrame([row])
    if os.path.exists(SUMMARY_FILE):
        old = pd.read_csv(SUMMARY_FILE)
        old = old[old["run"] != row["run"]]  # re-running a model replaces its old row
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(SUMMARY_FILE, index=False)


def evaluate_run(run_name, use_test=True):
    """Load the best checkpoint of a run and compute all metrics.

    use_test=False is for quick experiments: only validation numbers are computed,
    so the test fold is never used to choose between variants.
    """
    run_dir = os.path.join(RESULTS_DIR, run_name)
    with open(os.path.join(run_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(run_dir, "train_info.json")) as f:
        train_info = json.load(f)

    device = get_device()
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    loaders, _ = get_dataloaders(config["batch_size"])

    # 1) Validation: AUROC and threshold tuning.
    val_probs, val_labels = predict(model, loaders["val"], device)
    val_macro, val_per_class = auroc_scores(val_labels, val_probs)
    thresholds = tune_thresholds(val_labels, val_probs)
    np.save(os.path.join(run_dir, "val_probs.npy"), val_probs)

    metrics = {
        "run": run_name,
        "model": config["model"],
        **train_info,
        "val_macro_auroc": val_macro,
        "val_auroc_per_class": val_per_class,
        "thresholds": thresholds,
    }
    row = {"run": run_name, "model": config["model"], "params": train_info["params"],
           "train_time_s": train_info["train_time_s"], "best_epoch": train_info["best_epoch"],
           "val_macro_auroc": round(val_macro, 4)}
    print(f"[{run_name}] val macro AUROC: {val_macro:.4f}")

    # 2) Test: thresholds from validation are applied unchanged.
    if use_test:
        test_probs, test_labels = predict(model, loaders["test"], device)
        test_macro, test_per_class = auroc_scores(test_labels, test_probs)
        per_class = classification_metrics(test_labels, test_probs, thresholds)
        macro_f1 = float(np.mean([m["f1"] for m in per_class.values()]))
        np.save(os.path.join(run_dir, "test_probs.npy"), test_probs)

        metrics.update({"test_macro_auroc": test_macro, "test_auroc_per_class": test_per_class,
                        "test_macro_f1": macro_f1, "test_per_class": per_class})
        row["test_macro_auroc"] = round(test_macro, 4)
        for name in CLASSES:
            row[f"test_auroc_{name}"] = round(test_per_class[name], 4)
        row["test_macro_f1"] = round(macro_f1, 4)

        print(f"[{run_name}] TEST macro AUROC: {test_macro:.4f}, macro F1: {macro_f1:.4f}")
        for name in CLASSES:
            m = per_class[name]
            print(f"  {name:5s} AUROC {test_per_class[name]:.4f}  F1 {m['f1']:.3f}  "
                  f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  (threshold {thresholds[name]:.2f})")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    update_summary(row)
    return metrics


def write_final_table(runs):
    """Clean markdown table of the final models -> results/final_table.md."""
    names = {"cnn": "1D-CNN (baseline)", "transformer": "Patch Transformer",
             "hybrid": "Hybrid CNN + Transformer"}
    lines = ["# Final results on the PTB-XL test fold (fold 10)", "",
             "| Model | Parameters | Train time | Macro AUROC | "
             + " | ".join(f"AUROC {c}" for c in CLASSES) + " | Macro F1 |",
             "|---|---|---|---|" + "---|" * len(CLASSES) + "---|"]
    for run in runs:
        with open(os.path.join(RESULTS_DIR, run, "metrics.json")) as f:
            m = json.load(f)
        per_class = " | ".join(f"{m['test_auroc_per_class'][c]:.3f}" for c in CLASSES)
        lines.append(f"| {names.get(run, run)} | {m['params']:,} | {m['train_time_s']:.0f} s | "
                     f"**{m['test_macro_auroc']:.4f}** | {per_class} | {m['test_macro_f1']:.3f} |")
    lines += ["", "- Test fold = official PTB-XL fold 10 (2,158 ECGs), used once per model.",
              "- Thresholds for F1 were tuned per class on the validation fold (fold 9).",
              "- Train time: one run on an NVIDIA RTX 4050 laptop GPU, seed 42, with early stopping.",
              "- Published deep models reach about 0.93 macro AUROC on this task."]
    with open(os.path.join(RESULTS_DIR, "final_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run name to evaluate, e.g. cnn")
    parser.add_argument("--no-test", action="store_true", help="validation metrics only")
    parser.add_argument("--final-table", nargs="+", help="runs to put in results/final_table.md")
    args = parser.parse_args()
    if args.run:
        evaluate_run(args.run, use_test=not args.no_test)
    if args.final_table:
        write_final_table(args.final_table)
