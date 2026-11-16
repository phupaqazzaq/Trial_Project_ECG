"""Milestone 7: turn the model output for one ECG into a short plain-language report.

Run with:  python -m src.llm_report --config configs/llm.yaml

Three ways to write the report (chosen with `provider` in the config):
  qwen       a free Qwen model that runs on this computer (no account, no API key)
  anthropic  Claude through the API (needs the ANTHROPIC_API_KEY environment variable)
  template   fixed sentences, no LLM
Every LLM text is checked automatically (check_report). If the LLM is not available
or its text breaks a rule, the template is used instead, so the pipeline always runs.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

from src.data import CACHE_DIR, CLASSES, RESULTS_DIR
from src.dataset import load_splits
from src.explain import explain_ecg, load_trained_model, top_windows
from src.utils import SEED, get_device, set_seed

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
- First say what the model decided, as given in "model_decision". A class is predicted only \
if it is listed in "predicted_classes". If that list is empty, say that no class reached its \
threshold; do not call the ECG normal.
- Mention the confidence of each predicted class as given in "confidence", with its \
probability and threshold.
- Name the time segments (in seconds) that influenced the result most, as given in \
"top_attended_windows".
- Use only the information in the JSON. Do not explain what a condition means, how serious \
it is or what it suggests. Do not invent clinical details, symptoms, causes, measurements \
or treatment advice.
- The last sentence must be exactly: "{DISCLAIMER}"

Example. For an input with "patient_age": 61, "patient_sex": "male", \
"predicted_classes": ["STTC", "HYP"], "probabilities": {{"STTC": 0.91, "HYP": 0.44, ...}}, \
"decision_thresholds": {{"STTC": 0.36, "HYP": 0.2, ...}}, \
"model_decision": "The model predicts: ST/T change (STTC) with high confidence, hypertrophy \
(HYP) with low confidence.", \
"top_attended_windows": ["2.0-2.5 s", "7.5-8.0 s", "4.0-4.5 s"], a good report is:
For this 61-year-old male patient, the model predicts ST/T change (STTC) and hypertrophy (HYP). \
Its confidence is high for STTC, with a probability of 0.91 against a threshold of 0.36, and \
low for HYP, with a probability of 0.44 against a threshold of 0.2. The other classes stayed \
below their thresholds. The parts of the recording that influenced the result most were \
2.0-2.5 s, 7.5-8.0 s and 4.0-4.5 s. {DISCLAIMER}
"""

# Wording that usually starts a medical interpretation or advice. The report must only
# restate the model output, so an LLM text containing one of these is rejected.
INTERPRETATION_WORDS = ["suggest", "further", "investigat", "recommend", "should",
                        "consult", "doctor", "treatment", "evidence", "abnormal", "healthy",
                        "heart attack", "serious", "severe", "risk", "urgent"]


def build_case(ecg_id, age, sex, probs, thresholds, windows):
    """Collect everything the report is allowed to use for one ECG."""
    predicted = [c for c, p in zip(CLASSES, probs) if p >= thresholds[c]]
    # The decision spelled out in words, so a small LLM cannot misread an empty list.
    # confidence in words, from fixed cut-offs, so the LLM does not have to judge it
    confidence = {c: confidence_word(probs[CLASSES.index(c)]) for c in predicted}
    if predicted:
        decision = "The model predicts: " + ", ".join(
            f"{CLASS_NAMES[c]} ({c}) with {confidence[c]} confidence" for c in predicted) + "."
    else:
        decision = "No class reached its decision threshold, so the model predicts no class."
    return {
        "ecg_id": int(ecg_id),
        # PTB-XL stores ages above 89 as 300 for privacy
        "patient_age": "over 89" if age >= 300 else int(age),
        "patient_sex": "male" if sex == 0 else "female",
        "model_decision": decision,
        "predicted_classes": predicted,
        "class_meanings": CLASS_NAMES,
        "probabilities": {c: round(float(p), 2) for c, p in zip(CLASSES, probs)},
        "decision_thresholds": {c: round(thresholds[c], 2) for c in CLASSES},
        "confidence": confidence,
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


_QWEN = {}  # the loaded Qwen model is kept here, so it is loaded only once per program


def load_qwen(model_name):
    """Load a Qwen chat model from Hugging Face (downloaded once, then read from disk)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # only needed for this provider

    if model_name not in _QWEN:
        device = get_device()
        # half precision on the GPU: half the memory (about 3 GB for the 1.5B model)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        model.eval()
        _QWEN[model_name] = (tokenizer, model)
    return _QWEN[model_name]


def qwen_report(case, config):
    """Let the local Qwen model write the report. Same prompt and same input as for Claude."""
    import torch

    tokenizer, model = load_qwen(config["model"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(case, indent=2)}]
    # the chat template adds the special tokens that mark system / user / assistant turns
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # do_sample=False = always take the most likely next word, so the same input
        # gives the same report every time (reproducible)
        output = model.generate(**inputs, max_new_tokens=250, do_sample=False)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]  # drop the prompt, keep the answer
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip() or None


def allowed_numbers(case):
    """Every number a report may contain. All of them come from the inputs."""
    numbers = {10.0, 2.0, 3.0, 4.0, 5.0}  # 10-second recording; small counts ("the 3 segments")
    for value in list(case["probabilities"].values()) + list(case["decision_thresholds"].values()):
        numbers.update({value, 100 * value})  # written as 0.85 or as 85 (%)
    for window in case["top_attended_windows"]:  # e.g. "3.5-4.0 s"
        numbers.update(float(v) for v in window.replace(" s", "").split("-"))
    numbers.add(89.0 if case["patient_age"] == "over 89" else float(case["patient_age"]))
    numbers.add(float(case["ecg_id"]))
    return numbers


def check_report(text, case):
    """Automatic checks on an LLM report. Returns a list of problems (empty = accepted).

    A language model can ignore its instructions, so the rules that code can check
    are checked by code:
      1. it ends with the disclaimer
      2. it is one paragraph of 3 to 5 sentences
      3. every number in it appears in the inputs (no invented values)
      4. it names every predicted class
      5. it names at least one of the attended time windows
      6. its confidence words agree with the inputs
      7. it contains no wording that signals interpretation or advice
    """
    problems = []
    if not text.endswith(DISCLAIMER):
        problems.append("does not end with the disclaimer")

    if "\n" in text:
        problems.append("more than one paragraph (line breaks or a list)")
    # a sentence ends with . ! or ? followed by a space and a capital letter
    n_sentences = len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()))
    if not 3 <= n_sentences <= 5:
        problems.append(f"{n_sentences} sentences instead of 3-5")

    allowed = allowed_numbers(case)
    for number in re.findall(r"\d+(?:\.\d+)?", text):
        if not any(abs(float(number) - a) < 1e-6 for a in allowed):
            problems.append(f"number {number} is not in the inputs")

    for c in case["predicted_classes"]:
        if c not in text and CLASS_NAMES[c].lower() not in text.lower():
            problems.append(f"predicted class {c} is not mentioned")

    # at least one attended window must be named: both its start and its end appear
    found = set(re.findall(r"\d+(?:\.\d+)?", text))
    windows = [w.replace(" s", "").split("-") for w in case["top_attended_windows"]]
    if not any(start in found and end in found for start, end in windows):
        problems.append("no attended time window is named")

    body = text.replace(DISCLAIMER, "").lower()  # the disclaimer itself is always allowed

    # confidence: every level given in the inputs must be mentioned, and the text must
    # not claim "high" confidence when no predicted class has it
    mentioned = set(re.findall(r"\b(high|moderate|low)(?:ly)?\b", body))
    expected = set(case["confidence"].values())
    for level in sorted(expected - mentioned):
        problems.append(f"{level} confidence is not mentioned")
    if "high" in mentioned and "high" not in expected:
        problems.append("says 'high' although no predicted class has high confidence")

    for word in INTERPRETATION_WORDS:
        if word in body:
            problems.append(f"interpretation wording '{word}'")
    return problems


def claude_report(case, config):
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
    """Returns (report text, source). The template is the fallback in every failure case."""
    provider = config.get("provider")
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return template_report(case), "template (no API key set)"
    if provider not in ("qwen", "anthropic"):
        return template_report(case), "template (no LLM chosen in the config)"

    try:
        text = qwen_report(case, config) if provider == "qwen" else claude_report(case, config)
    except Exception as error:  # e.g. model not downloaded, no internet: keep the pipeline running
        print(f"LLM call failed ({type(error).__name__}: {error}); using the template instead.")
        return template_report(case), f"template (LLM call failed: {type(error).__name__})"
    if not text:
        return template_report(case), "template (the LLM returned no text)"

    problems = check_report(text, case)
    if problems:
        print(f"LLM report for ECG {case['ecg_id']} rejected ({'; '.join(problems)}):\n  {text}")
        return template_report(case), "template (LLM text rejected: " + "; ".join(problems) + ")"
    return text, f"LLM ({config['model']}), passed the automatic checks"


def pick_report_examples(labels, probs, thresholds):
    """One correctly predicted test ECG per class (the first one not used yet), 5 in total."""
    chosen = []
    for c, name in enumerate(CLASSES):
        correct = np.where((labels[:, c] == 1) & (probs[:, c] >= thresholds[name]))[0]
        correct = [int(i) for i in correct if int(i) not in chosen]
        if correct:
            chosen.append(correct[0])
    return chosen


def acceptance_check(model, val, meta, thresholds, config, device, n):
    """How often does the LLM text pass check_report? Measured on n random VALIDATION ECGs
    (the prompt was also developed on validation ECGs, never on the test fold)."""
    rng = np.random.default_rng(SEED)
    accepted, rejected = 0, []
    for idx in rng.choice(len(val["x"]), n, replace=False):
        ecg_id = val["ecg_ids"][idx]
        probs, importance = explain_ecg(model, val["x"][idx], device)
        case = build_case(ecg_id, meta.loc[ecg_id, "age"], meta.loc[ecg_id, "sex"],
                          probs, thresholds, top_windows(importance))
        _, source = generate_report(case, config)
        if source.startswith("LLM"):
            accepted += 1
        else:
            rejected.append({"ecg_id": int(ecg_id), "source": source})
    result = {"model": config["model"], "n_validation_ecgs": n, "accepted": accepted,
              "accepted_share": round(accepted / n, 3), "rejected": rejected}
    with open(os.path.join(RESULTS_DIR, "llm_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nLLM reports accepted by the automatic checks: {accepted} of {n} "
          f"({100 * accepted / n:.0f}%). Saved {RESULTS_DIR}/llm_check.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm.yaml")
    parser.add_argument("--check-val", type=int, default=0, metavar="N",
                        help="instead of writing the examples, measure on N validation ECGs "
                             "how often the LLM text passes the automatic checks")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed()
    device = get_device()
    run_name = config["run"]
    model = load_trained_model(run_name, device)
    splits = load_splits()
    test = splits["test"]
    meta = pd.read_csv(os.path.join(CACHE_DIR, "meta.csv"), index_col="ecg_id")
    with open(os.path.join(RESULTS_DIR, run_name, "metrics.json")) as f:
        thresholds = json.load(f)["thresholds"]
    if args.check_val:
        acceptance_check(model, splits["val"], meta, thresholds, config, device, args.check_val)
        return
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
