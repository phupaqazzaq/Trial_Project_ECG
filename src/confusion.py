"""Confusion matrices (TP / FP / FN / TN) for every model and class on the test fold.

Run with:  python -m src.confusion --runs cnn transformer hybrid

Nothing is trained or tuned here. The script reuses what training already saved:
  results/<run>/test_probs.npy   predicted probabilities on fold 10
  results/<run>/metrics.json     per-class thresholds, tuned on the validation fold (fold 9)

The task is multi-label (one ECG can be MI and CD at once), so there is no single 5x5
confusion matrix. Each class is its own yes/no question with its own 2x2 table.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # draw figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CLASSES, FIG_DIR, RESULTS_DIR, load_cache, split_masks

NAMES = {"cnn": "1D-CNN (baseline)", "transformer": "Patch Transformer",
         "hybrid": "Hybrid CNN + Transformer"}


def test_labels():
    """True multi-hot labels of the test fold, in the same order as test_probs.npy."""
    data = load_cache()
    return data["Y"][split_masks(data["folds"])["test"]].astype(bool)


def confusion_counts(y_true, probs, threshold):
    """TP, FP, FN, TN for one class. A class is predicted when probability >= threshold."""
    pred = probs >= threshold
    tp = int((pred & y_true).sum())    # has it, model says yes
    fp = int((pred & ~y_true).sum())   # does not have it, model says yes (false alarm)
    fn = int((~pred & y_true).sum())   # has it, model says no (missed)
    tn = int((~pred & ~y_true).sum())  # does not have it, model says no
    return tp, fp, fn, tn


def run_table(run, y):
    """One row per class with the four counts and the metrics built from them."""
    probs = np.load(os.path.join(RESULTS_DIR, run, "test_probs.npy"))
    with open(os.path.join(RESULTS_DIR, run, "metrics.json")) as f:
        thresholds = json.load(f)["thresholds"]
    rows = []
    for i, name in enumerate(CLASSES):
        tp, fp, fn, tn = confusion_counts(y[:, i], probs[:, i], thresholds[name])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"run": run, "class": name, "threshold": round(thresholds[name], 2),
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)})
    # ECGs where all 5 yes/no answers are right (a much stricter score than per-class F1)
    pred_all = probs >= np.array([thresholds[c] for c in CLASSES])
    exact = int((pred_all == y).all(axis=1).sum())
    return pd.DataFrame(rows), exact


def write_markdown(tables, exact, n_ecgs):
    lines = ["# Confusion matrices on the PTB-XL test fold (fold 10)", "",
             f"{n_ecgs:,} test ECGs. Each class uses the threshold tuned on the validation fold.",
             "TP = has it, predicted yes. FP = does not have it, predicted yes (false alarm).",
             "FN = has it, predicted no (missed). TN = does not have it, predicted no.", ""]
    for run, table in tables.items():
        lines += [f"## {NAMES.get(run, run)}", "",
                  "| Class | Threshold | TP | FP | FN | TN | Precision | Recall | F1 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in table.itertuples():
            lines.append(f"| {r[2]} | {r.threshold:.2f} | {r.TP:,} | {r.FP:,} | {r.FN:,} | {r.TN:,} | "
                         f"{r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} |")
        lines += ["", f"Macro F1 {table['f1'].mean():.3f}. All 5 answers right for "
                      f"{exact[run]:,} of {n_ecgs:,} ECGs ({100 * exact[run] / n_ecgs:.1f} %).", ""]
    with open(os.path.join(RESULTS_DIR, "confusion_matrix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot(tables):
    """Grid of 2x2 tables: one row per model, one column per class.
    Colour = share of the true row (so rare and common classes are comparable)."""
    runs = list(tables)
    fig, axes = plt.subplots(len(runs), len(CLASSES), figsize=(2.6 * len(CLASSES), 2.6 * len(runs)))
    axes = np.atleast_2d(axes)
    for r, run in enumerate(runs):
        for c, name in enumerate(CLASSES):
            row = tables[run].iloc[c]
            counts = np.array([[row.TP, row.FN], [row.FP, row.TN]])  # rows: true yes / true no
            share = counts / counts.sum(axis=1, keepdims=True)
            ax = axes[r, c]
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(counts):
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else "black", fontsize=11)
            ax.set_xticks([0, 1], ["pred yes", "pred no"], fontsize=8)
            ax.set_yticks([0, 1], ["true yes", "true no"], fontsize=8)
            if r == 0:
                ax.set_title(name, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(NAMES.get(run, run) + "\n\ntrue yes / true no", fontsize=9)
            ax.set_xlabel(f"F1 {row.f1:.3f}", fontsize=9)
    fig.suptitle("Confusion matrices per class, test fold (fold 10)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["cnn", "transformer", "hybrid"])
    args = parser.parse_args()

    y = test_labels()
    tables, exact = {}, {}
    for run in args.runs:
        tables[run], exact[run] = run_table(run, y)
        print(f"\n{NAMES.get(run, run)}  (all 5 right: {exact[run]} / {len(y)})")
        print(tables[run].drop(columns="run").to_string(index=False))

    pd.concat(tables.values()).to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"), index=False)
    write_markdown(tables, exact, len(y))
    plot(tables)
    print("\nSaved results/confusion_matrix.md, results/confusion_matrix.csv, "
          "figures/confusion_matrices.png")


if __name__ == "__main__":
    main()
