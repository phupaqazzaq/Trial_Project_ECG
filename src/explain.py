"""Milestone 6: attention rollout, attention plots, ROC curves, model comparison.

Run with:  python -m src.explain --run transformer --roc-run cnn --compare cnn transformer hybrid
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_curve

from src.data import CLASSES, FIG_DIR, LEADS, RESULTS_DIR
from src.dataset import load_splits
from src.models import build_model
from src.preprocess import STATS_FILE
from src.utils import get_device, set_seed

WINDOW = 50  # 50 time steps = 0.5 s, the size of one reported time window


def load_trained_model(run_name, device):
    """Rebuild a model from its saved config and load the best checkpoint."""
    run_dir = os.path.join(RESULTS_DIR, run_name)
    with open(os.path.join(run_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    model.eval()
    return model


def attention_rollout(attentions):
    """Combine the attention of all layers into one importance score per token.

    attentions: list (one per layer) of tensors (heads, tokens, tokens) for ONE ECG.
    Steps:
      1. average over the heads
      2. add the identity matrix (the residual connection lets each token keep its own
         information) and re-normalize so every row sums to 1
      3. multiply the matrices of all layers -> how much each input token flows into
         each output token through the whole network
      4. read the [CLS] row: how much each patch contributed to the final decision
    """
    n_tokens = attentions[0].shape[-1]
    rollout = torch.eye(n_tokens)
    for attn in attentions:
        a = attn.mean(dim=0)                      # 1. average heads -> (tokens, tokens)
        a = 0.5 * a + 0.5 * torch.eye(n_tokens)   # 2. residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout                     # 3. chain the layers
    return rollout[0, 1:].numpy()                 # 4. [CLS] row, without [CLS] itself


@torch.no_grad()
def explain_ecg(model, x, device):
    """x: one standardized ECG, array (1000, 12).

    Returns (probabilities (5,), importance per time step (1000,)).
    The importance is scaled so that it sums to 1 over the 10 seconds.
    """
    inp = torch.from_numpy(x.T.copy()).unsqueeze(0).to(device)  # (1, 12, 1000)
    logits, attentions = model(inp, return_attention=True)
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()
    token_importance = attention_rollout([a[0].float().cpu() for a in attentions])
    # Each token covers the same number of time steps (50 for the patch Transformer).
    steps_per_token = x.shape[0] // len(token_importance)
    per_step = np.repeat(token_importance, steps_per_token) / steps_per_token
    return probs, per_step / per_step.sum()


def top_windows(per_step_importance, k=3):
    """The k most important 0.5 s windows, as (start_s, end_s, share_of_attention)."""
    window_scores = per_step_importance.reshape(-1, WINDOW).sum(axis=1)  # 20 windows
    best = np.argsort(window_scores)[::-1][:k]
    return [(i * WINDOW / 100, (i + 1) * WINDOW / 100, float(window_scores[i])) for i in best]


def draw_attention(x_mv, importance, title, leads=("II", "V2")):
    """ECG (in mV) with the attention drawn as red shading, plus a bar chart per 0.5 s window.
    Returns the matplotlib figure (the web app shows it, plot_attention saves it)."""
    time = np.arange(len(x_mv)) / 100
    shade = importance / importance.max()  # strongest token = full colour
    window_share = 100 * importance.reshape(-1, WINDOW).sum(axis=1)  # % of attention per 0.5 s

    fig, axes = plt.subplots(len(leads) + 1, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3] * len(leads) + [1.6]})
    for ax, lead in zip(axes, leads):
        ax.plot(time, x_mv[:, LEADS.index(lead)], color="black", linewidth=0.8)
        low, high = ax.get_ylim()
        # one coloured column per time step, stretched over the full height of the plot
        ax.imshow(shade[None, :], extent=[0, 10, low, high], aspect="auto",
                  cmap="Reds", vmin=0, vmax=1, alpha=0.55, zorder=0)
        ax.set_ylim(low, high)
        ax.set_ylabel(f"Lead {lead} (mV)")
        ax.grid(alpha=0.25)

    # Bottom panel: the same information as numbers. If the model looked at all
    # windows equally, every bar would be at the dashed line (100% / 20 windows = 5%).
    bars = axes[-1]
    bars.bar(np.arange(len(window_share)) * 0.5 + 0.25, window_share, width=0.45, color="tab:red")
    bars.axhline(100 / len(window_share), color="gray", linestyle="--", linewidth=1,
                 label="equal attention (5%)")
    bars.set_ylabel("Attention (%)")
    bars.legend(loc="upper left", fontsize=8)
    bars.set_xlim(0, 10)
    bars.set_xticks(np.arange(0, 10.5, 0.5))
    bars.set_xlabel("Time (s)  -  darker red = more attention (rollout from [CLS])")
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_attention(x_mv, importance, title, filename, leads=("II", "V2")):
    """Draw the attention figure and save it to figures/."""
    fig = draw_attention(x_mv, importance, title, leads)
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close(fig)


def attention_statistics(model, x_all, device, run_name):
    """Numbers that describe the attention over the WHOLE test fold (not just a few plots).

    - top_window_share: how much of the attention the strongest 0.5 s window gets (5% = equal)
    - energy_correlation: Spearman correlation between a window's attention and its signal
      energy (mean absolute amplitude). Positive = the model looks more at windows with
      big deflections (QRS complexes) than at flat baseline.
    - position_profile: average attention per window position (shows position bias).
    """
    from scipy.stats import spearmanr

    top_shares, correlations, profiles = [], [], []
    for x in x_all:
        _, importance = explain_ecg(model, x, device)
        window_share = importance.reshape(-1, WINDOW).sum(axis=1)
        window_energy = np.abs(x).reshape(-1, WINDOW, x.shape[1]).mean(axis=(1, 2))
        top_shares.append(window_share.max())
        correlations.append(spearmanr(window_share, window_energy)[0])
        profiles.append(window_share)

    stats = {
        "n_ecgs": len(x_all),
        "mean_top_window_share": float(np.mean(top_shares)),
        "mean_energy_correlation": float(np.nanmean(correlations)),
        "share_of_ecgs_with_positive_correlation": float(np.mean(np.array(correlations) > 0)),
        "position_profile": [round(float(v), 4) for v in np.mean(profiles, axis=0)],
    }
    with open(os.path.join(RESULTS_DIR, f"attention_stats_{run_name}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Attention statistics over {stats['n_ecgs']} test ECGs:")
    print(f"  strongest window gets on average {100 * stats['mean_top_window_share']:.1f}% (equal = 5%)")
    print(f"  mean Spearman correlation attention vs. signal energy: {stats['mean_energy_correlation']:+.3f} "
          f"(positive for {100 * stats['share_of_ecgs_with_positive_correlation']:.0f}% of ECGs)")
    print(f"  average attention per window position (%): "
          f"{[round(100 * v, 1) for v in stats['position_profile']]}")
    return stats


def describe(labels, probs, thresholds):
    """Short text for figure titles: true labels and predicted probabilities."""
    true = "+".join(c for c, v in zip(CLASSES, labels) if v == 1)
    pred = ", ".join(f"{c} {p:.2f}{'*' if p >= thresholds[c] else ''}" for c, p in zip(CLASSES, probs))
    return f"true: {true} | predicted: {pred}   (* = above threshold)"


def pick_examples(labels, probs, thresholds):
    """Choose which test ECGs to plot.

    Correct: for MI, STTC, CD the 2 most confident true positives that have only that label.
    Wrong:   the most confident missed MI (false negative) and the normal ECG with the
             highest MI probability (false alarm).
    """
    single_label = labels.sum(axis=1) == 1
    chosen = []
    for name in ["MI", "STTC", "CD"]:
        c = CLASSES.index(name)
        correct = np.where(single_label & (labels[:, c] == 1) & (probs[:, c] >= thresholds[name]))[0]
        for idx in correct[np.argsort(probs[correct, c])[::-1][:2]]:
            chosen.append((name, int(idx)))

    mi = CLASSES.index("MI")
    missed = np.where(single_label & (labels[:, mi] == 1) & (probs[:, mi] < thresholds["MI"]))[0]
    if len(missed):
        chosen.append(("misclassified_missedMI", int(missed[np.argmin(probs[missed, mi])])))
    normal = np.where(single_label & (labels[:, 0] == 1) & (probs[:, mi] >= thresholds["MI"]))[0]
    if len(normal):
        chosen.append(("misclassified_falseMI", int(normal[np.argmax(probs[normal, mi])])))
    return chosen


def plot_roc_curves(labels, probs, run_name):
    """One ROC curve per class on the test set."""
    fig, ax = plt.subplots(figsize=(6, 6))
    with open(os.path.join(RESULTS_DIR, run_name, "metrics.json")) as f:
        aurocs = json.load(f)["test_auroc_per_class"]
    for i, name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUROC {aurocs[name]:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="random guess (0.5)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC curves on the test fold - {run_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)


def plot_model_comparison(runs):
    """Bar chart of test macro AUROC for the main models."""
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv")).set_index("run")
    runs = [r for r in runs if r in summary.index and not np.isnan(summary.loc[r, "test_macro_auroc"])]
    scores = [summary.loc[r, "test_macro_auroc"] for r in runs]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(runs, scores, color=["tab:gray", "tab:blue", "tab:green"][:len(runs)])
    ax.bar_label(bars, fmt="%.4f", label_type="center", color="white", fontweight="bold")
    ax.axhline(0.93, color="red", linestyle="--", linewidth=1, label="published deep models (~0.93)")
    ax.set_ylim(0.80, 0.96)
    ax.set_ylabel("Test macro AUROC")
    ax.set_title("Model comparison on the PTB-XL test fold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="transformer", help="run whose attention is explained")
    parser.add_argument("--prefix", default="attention", help="start of the figure file names")
    parser.add_argument("--roc-run", default=None, help="best model: draw its ROC curves")
    parser.add_argument("--compare", nargs="+", default=None, help="runs for the comparison bar chart")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    model = load_trained_model(args.run, device)
    test = load_splits()["test"]
    stats = np.load(STATS_FILE)
    with open(os.path.join(RESULTS_DIR, args.run, "metrics.json")) as f:
        thresholds = json.load(f)["thresholds"]
    probs = np.load(os.path.join(RESULTS_DIR, args.run, "test_probs.npy"))
    labels = test["y"]

    rows = []
    for tag, idx in pick_examples(labels, probs, thresholds):
        ecg_id = int(test["ecg_ids"][idx])
        ecg_probs, importance = explain_ecg(model, test["x"][idx], device)
        x_mv = test["x"][idx] * stats["std"] + stats["mean"]  # undo standardization for plotting
        filename = f"{args.prefix}_{tag}_{ecg_id}.png"
        plot_attention(x_mv, importance,
                       f"{args.run} | ecg_id {ecg_id} | {describe(labels[idx], ecg_probs, thresholds)}",
                       filename)
        windows = "; ".join(f"{a:.1f}-{b:.1f}s ({100 * s:.0f}%)" for a, b, s in top_windows(importance))
        rows.append({"figure": filename, "example": tag, "ecg_id": ecg_id, "top_windows": windows})
        print(f"{filename}: top windows {windows}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f"{args.prefix}_examples_{args.run}.csv"),
                              index=False)

    attention_statistics(model, test["x"], device, args.run)

    if args.roc_run:
        roc_probs = np.load(os.path.join(RESULTS_DIR, args.roc_run, "test_probs.npy"))
        plot_roc_curves(labels, roc_probs, args.roc_run)
    if args.compare:
        plot_model_comparison(args.compare)


if __name__ == "__main__":
    main()
