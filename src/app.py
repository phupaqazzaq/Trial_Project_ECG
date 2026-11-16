"""Local test website: pick a test ECG, see the prediction, the attention map and the report.

Run with:  python -m src.app      (then open http://127.0.0.1:7860 in a browser)

Nothing is trained here and nothing is sent to the internet. The page only reuses
functions from the other files:
  explain.py     -> prediction + attention rollout + the figure
  llm_report.py  -> the plain-language report (local Qwen model, with a template fallback)
"""

import argparse
import json
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data import CACHE_DIR, CLASSES, RESULTS_DIR
from src.dataset import load_splits
from src.explain import describe, draw_attention, explain_ecg, load_trained_model, top_windows
from src.llm_report import CLASS_NAMES, DISCLAIMER, build_case, generate_report, load_qwen
from src.preprocess import STATS_FILE
from src.utils import get_device, set_seed

# ---- Load everything once, when the program starts ----
set_seed()
with open("configs/llm.yaml") as f:
    LLM_CONFIG = yaml.safe_load(f)
RUN = LLM_CONFIG["run"]  # the model with attention (hybrid)
DEVICE = get_device()
MODEL = load_trained_model(RUN, DEVICE)
TEST = load_splits()["test"]  # fold 10: ECGs the model has never seen in training
STATS = np.load(STATS_FILE)
META = pd.read_csv(os.path.join(CACHE_DIR, "meta.csv"), index_col="ecg_id")
with open(os.path.join(RESULTS_DIR, RUN, "metrics.json")) as f:
    THRESHOLDS = json.load(f)["thresholds"]


def true_labels(i):
    return ", ".join(c for c, flag in zip(CLASSES, TEST["y"][i]) if flag == 1)


# One line per test ECG for the drop-down list, e.g. "ECG 162 (true: MI, CD)".
# Typing in the drop-down filters the list, so typing "HYP" shows the HYP ECGs.
CHOICES = [f"ECG {ecg_id} (true: {true_labels(i)})" for i, ecg_id in enumerate(TEST["ecg_ids"])]


def analyze(choice):
    """Everything the page shows for one ECG."""
    i = CHOICES.index(choice)
    ecg_id = int(TEST["ecg_ids"][i])
    x = TEST["x"][i]  # standardized signal (1000, 12), exactly what the model was tested on

    # 1. prediction + attention (one forward pass)
    probs, importance = explain_ecg(MODEL, x, DEVICE)

    # 2. table: probability against the threshold that was tuned on the validation fold
    table = pd.DataFrame({
        "Class": CLASSES,
        "Meaning": [CLASS_NAMES[c] for c in CLASSES],
        "Probability": [round(float(p), 2) for p in probs],
        "Threshold": [round(THRESHOLDS[c], 2) for c in CLASSES],
        "Predicted": ["yes" if p >= THRESHOLDS[c] else "no" for c, p in zip(CLASSES, probs)],
        "True label": ["yes" if flag == 1 else "no" for flag in TEST["y"][i]],
    })

    # 3. attention figure (signal converted back to millivolts for plotting)
    x_mv = x * STATS["std"] + STATS["mean"]
    figure = draw_attention(x_mv, importance,
                            f"{RUN} | ecg_id {ecg_id} | {describe(TEST['y'][i], probs, THRESHOLDS)}")
    plt.close(figure)  # the page keeps its own copy; closing frees matplotlib's memory

    # 4. report: only these facts are given to the language model
    case = build_case(ecg_id, META.loc[ecg_id, "age"], META.loc[ecg_id, "sex"],
                      probs, THRESHOLDS, top_windows(importance))
    report, source = generate_report(case, LLM_CONFIG)
    report_text = f"{report}\n\n*Source: {source}*"
    return table, figure, report_text, case


def random_choice():
    return CHOICES[np.random.randint(len(CHOICES))]


with gr.Blocks(title="ECG demo") as demo:
    gr.Markdown(
        "# ECG disease prediction demo\n"
        f"Model: **{RUN}** (CNN + Transformer), trained on PTB-XL folds 1-8. "
        f"The {len(CHOICES):,} ECGs below are the official test fold (fold 10).\n\n"
        f"**{DISCLAIMER}**"
    )
    with gr.Row():
        ecg_choice = gr.Dropdown(CHOICES, value=CHOICES[0], label="Test ECG (type to search, e.g. HYP)",
                                 scale=4)
        random_button = gr.Button("Random ECG", scale=1)
        analyze_button = gr.Button("Analyze", variant="primary", scale=1)

    gr.Markdown("## 1. Prediction")
    table_out = gr.Dataframe(label="A class is predicted when its probability reaches its threshold")
    gr.Markdown("## 2. Where the model looked (attention rollout, darker red = more attention)")
    figure_out = gr.Plot(show_label=False)
    gr.Markdown("## 3. Plain-language report")
    report_out = gr.Markdown()
    gr.Markdown("## 4. The only facts the report generator received")
    case_out = gr.JSON(label="Input of the report generator")

    outputs = [table_out, figure_out, report_out, case_out]
    analyze_button.click(analyze, inputs=ecg_choice, outputs=outputs)
    # "Random ECG" first picks an ECG, then analyzes it
    random_button.click(random_choice, outputs=ecg_choice).then(analyze, inputs=ecg_choice,
                                                                 outputs=outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    if LLM_CONFIG.get("provider") == "qwen":
        # load the language model now, so the first click is not slow
        try:
            load_qwen(LLM_CONFIG["model"])
        except Exception as error:
            print(f"Qwen could not be loaded ({type(error).__name__}); reports will use the template.")
    demo.launch(server_port=args.port, inbrowser=not args.no_browser)
