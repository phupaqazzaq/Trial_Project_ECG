"""Milestone 7: turn the model output for one ECG into a short plain-language report.

Run with:  python -m src.llm_report --config configs/llm.yaml

If the API key environment variable is not set, a fixed template is used instead,
so the pipeline always runs.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from src.data import CACHE_DIR, CLASSES, RESULTS_DIR
from src.dataset import load_splits
from src.explain import explain_ecg, load_trained_model, top_windows
from src.utils import get_device, set_seed

CLASS_NAMES = {
    "NORM": "normal ECG",
    "MI": "myocardial infarction",
    "STTC": "ST/T change",
    "CD": "conduction disturbance",
    "HYP": "hypertrophy",
}
DISCLAIMER = "This is an AI research prototype, not a medical diagnosis."

SYSTEM_PROMPT = f"""You write short reports that explain the output of an ECG classification model \
to a non-expert reader. You receive a JSON object with the model's output for one ECG.

Rules:
- Use plain language. Write one paragraph of 3 to 5 sentences. No headings, no bullet points.
- Say which condition(s) the model predicted and mention the confidence, using the given \
probabilities and thresholds.
- Name the time segments (in seconds) that influenced the result most, as given in \
"top_attended_windows".
- Use only the information in the JSON. Do not invent clinical details, symptoms, causes, \
measurements or treatment advice.
- The last sentence must be exactly: "{DISCLAIMER}"
"""


def build_case(ecg_id, age, sex, probs, thresholds, windows):
    """Collect everything the report is allowed to use for one ECG."""
    predicted = [c for c, p in zip(CLASSES, probs) if p >= thresholds[c]]
    return {
        "ecg_id": int(ecg_id),
        # PTB-XL stores ages above 89 as 300 for privacy
        "patient_age": "over 89" if age >= 300 else int(age),
        "patient_sex": "male" if sex == 0 else "female",
        "predicted_classes": predicted,
        "class_meanings": CLASS_NAMES,
        "probabilities": {c: round(float(p), 2) for c, p in zip(CLASSES, probs)},
        "decision_thresholds": {c: round(thresholds[c], 2) for c in CLASSES},
        "top_attended_windows": [f"{a:.1f}-{b:.1f} s" for a, b, _ in windows],
    }


def confidence_word(prob):
    if prob >= 0.8:
        return "high"
    if prob >= 0.5:
        return "moderate"
    return "low"


def template_report(case):
    """Fallback without an LLM: fill the same information into fixed sentences."""
    who = f"{case['patient_sex']} patient, age {case['patient_age']}"
    probs, thresholds = case["probabilities"], case["decision_thresholds"]

    if case["predicted_classes"]:
        findings = [f"{CLASS_NAMES[c]} (probability {probs[c]:.2f}, threshold {thresholds[c]:.2f}, "
                    f"{confidence_word(probs[c])} confidence)" for c in case["predicted_classes"]]
        first = f"For this ECG ({who}), the model predicts: " + "; ".join(findings) + "."
    else:
        top = max(probs, key=probs.get)
        first = (f"For this ECG ({who}), no class reached its decision threshold; the highest "
                 f"score was {CLASS_NAMES[top]} with probability {probs[top]:.2f}.")

    others = [c for c in CLASSES if c not in case["predicted_classes"]]
    second = ("The other classes stayed below their thresholds ("
              + ", ".join(f"{c} {probs[c]:.2f}" for c in others) + ").") if others else ""
    third = ("The time segments that influenced the result most were "
             + ", ".join(case["top_attended_windows"]) + " of the 10-second recording.")
    return " ".join(s for s in [first, second, third, DISCLAIMER] if s)


def llm_report(case, config):
    """Ask Claude to write the report. The API key is read from the environment by the SDK."""
    import anthropic  # imported here so the template fallback works without the package

    client = anthropic.Anthropic()  # uses the ANTHROPIC_API_KEY environment variable
    response = client.messages.create(
        model=config["model"],
        max_tokens=4000,
        output_config={"effort": "low"},  # a short summary does not need deep reasoning
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(case, indent=2)}],
    )
    if response.stop_reason == "refusal":
        return None
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or None


def generate_report(case, config):
    """Returns (report text, source). Falls back to the template if no key or the call fails."""
    if config.get("provider") == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            text = llm_report(case, config)
            if text:
                return text, f"LLM ({config['model']})"
        except Exception as error:  # e.g. no internet, invalid key: keep the pipeline running
            print(f"LLM call failed ({type(error).__name__}: {error}); using the template instead.")
    return template_report(case), "template (no API key set)"


def pick_report_examples(labels, probs, thresholds):
    """One correctly predicted test ECG per class (the first one not used yet), 5 in total."""
    chosen = []
    for c, name in enumerate(CLASSES):
        correct = np.where((labels[:, c] == 1) & (probs[:, c] >= thresholds[name]))[0]
        correct = [int(i) for i in correct if int(i) not in chosen]
        if correct:
            chosen.append(correct[0])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed()
    device = get_device()
    run_name = config["run"]
    model = load_trained_model(run_name, device)
    test = load_splits()["test"]
    meta = pd.read_csv(os.path.join(CACHE_DIR, "meta.csv"), index_col="ecg_id")
    with open(os.path.join(RESULTS_DIR, run_name, "metrics.json")) as f:
        thresholds = json.load(f)["thresholds"]
    test_probs = np.load(os.path.join(RESULTS_DIR, run_name, "test_probs.npy"))

    lines = ["# Example ECG reports", "",
             f"Model: `{run_name}`. Each report is generated only from the inputs listed above it.", ""]
    for idx in pick_report_examples(test["y"], test_probs, thresholds):
        ecg_id = test["ecg_ids"][idx]
        probs, importance = explain_ecg(model, test["x"][idx], device)
        case = build_case(ecg_id, meta.loc[ecg_id, "age"], meta.loc[ecg_id, "sex"],
                          probs, thresholds, top_windows(importance))
        report, source = generate_report(case, config)
        true_labels = ", ".join(c for c, v in zip(CLASSES, test["y"][idx]) if v == 1)

        lines += [f"## ECG {case['ecg_id']} (true labels: {true_labels})", "",
                  "**Inputs given to the report generator**", "",
                  f"- Patient: {case['patient_age']}, {case['patient_sex']}",
                  f"- Predicted classes: {', '.join(case['predicted_classes']) or 'none'}",
                  "- Probabilities: " + ", ".join(f"{c} {p:.2f}" for c, p in case["probabilities"].items()),
                  "- Thresholds: " + ", ".join(f"{c} {t:.2f}" for c, t in case["decision_thresholds"].items()),
                  f"- Top attended windows: {', '.join(case['top_attended_windows'])}", "",
                  f"**Report** (source: {source})", "", f"> {report}", ""]
        print(f"ECG {case['ecg_id']}: {report}\n")

    with open(os.path.join(RESULTS_DIR, "llm_reports.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {RESULTS_DIR}/llm_reports.md")


if __name__ == "__main__":
    main()
