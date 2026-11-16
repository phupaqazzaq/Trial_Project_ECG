# ECG Disease Prediction: Patch Transformer + LLM Report

Task spec for Claude Code. Owner: Natdanai (MUICT). This is my **individual** trial project for Aj. Tipajin's senior project. She will question me on every step, so I must understand and be able to explain everything that gets built.

---

## 0. How to work with me (read first)

- **Work milestone by milestone** (Section 6). After each milestone, STOP and give me:
  1. what you built and the files changed,
  2. the results or outputs I should look at,
  3. a short "how to explain this to my professor" note (2–5 sentences, plain language).
  Wait for my "go" before starting the next milestone.
- Append each explanation note to `docs/explanations.md` so I end up with a study guide.
- **Keep code simple and readable.** Plain PyTorch, short functions, comments explaining *why*, no clever abstractions. I need to walk through it line by line.
- **Never tune anything on the test fold (fold 10).** Use it once at the end for final numbers.
- Set random seeds everywhere (`seed = 42`) so results are reproducible.
- If something fails or results look wrong, tell me plainly and propose a fix. Don't silently change the plan.
- Ask me before installing anything large or changing the method.

---

## 1. Goal

Predict heart disease from 12-lead ECG time-series data, then explain the prediction:

1. A **1D-CNN baseline** (simple reference model).
2. A **patch-based Transformer** (main model).
3. A **hybrid CNN + Transformer** (only if the pure Transformer underperforms).
4. **Explainability:** attention maps showing which time segments the model focused on.
5. **LLM report:** a short plain-language report generated from the model's output.
6. *(Optional bonus)* an **autoencoder anomaly detector** trained on normal ECGs only.

Deliverables: clean code, a results table, figures for slides, and a presentation notebook.

---

## 2. Environment

- Python 3.10+, PyTorch. Detect device automatically: CUDA → MPS (Apple) → CPU.
- Use mixed precision (`torch.autocast`) only when on CUDA.
- Packages: `torch numpy pandas scipy scikit-learn wfdb matplotlib tqdm pyyaml`. LLM step: `anthropic` or `openai` (see Milestone 7).
- Must also run on **Google Colab (free T4 GPU)**. Keep memory use modest and avoid anything that needs more than ~12 GB RAM.

---

## 3. Dataset: PTB-XL v1.0.3

- Source: https://physionet.org/content/ptb-xl/1.0.3/ (open access, no account needed).
- Download (only once, into `data/ptb-xl/`, and skip if already present):
  ```bash
  wget -r -N -c -np -nH --cut-dirs=3 -P data/ptb-xl https://physionet.org/files/ptb-xl/1.0.3/
  ```
  If wget is unavailable, download the zip from the PhysioNet page and unzip into `data/ptb-xl/`.
- 21,799 ECGs, 10 s each, 12 leads. Use the **100 Hz** version: the `filename_lr` column, loaded with `wfdb.rdsamp()`. Each ECG → array of shape **(1000, 12)**.
- After loading once, **cache** signals and labels to `data/cache/*.npy` (float32) so later runs are fast.

### Labels: 5 diagnostic superclasses (multi-label)

| Label | Meaning |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial infarction |
| STTC | ST/T change |
| CD | Conduction disturbance |
| HYP | Hypertrophy |

Mapping (the standard PTB-XL approach):
1. Read `ptbxl_database.csv` (`index_col='ecg_id'`) and parse `scp_codes` with `ast.literal_eval` (it's a dict of code → likelihood).
2. Read `scp_statements.csv` (`index_col=0`) and keep rows where `diagnostic == 1`.
3. For each ECG, collect the `diagnostic_class` of every code that appears in the filtered table → a set of superclasses.
4. **Drop ECGs with an empty set.** Turn labels into a multi-hot vector of length 5 in the order `[NORM, MI, STTC, CD, HYP]`.
5. Print and save the class counts (they're imbalanced; HYP is the rarest).

### Split (official, do not change)

Use the `strat_fold` column:
- Train: folds 1–8
- Validation: fold 9
- Test: fold 10

These folds keep each patient within a single fold, so no patient leaks between train and test. This also makes our results comparable with published papers.

---

## 4. Preprocessing

1. **Bandpass filter 0.5–40 Hz** (Butterworth, order 3, `scipy.signal.butter` + `filtfilt`, fs = 100). This removes baseline drift and high-frequency noise.
2. **Standardize per lead**: compute mean and std of each lead on the **training set only**, then apply them to train, val and test. Save the stats to `data/cache/norm_stats.npz`.
3. No data augmentation in the first version. *(Later, optional: small random scaling or noise on train only.)*

---

## 5. Project structure

```
ecg-transformer/
├── CLAUDE.md              # this file
├── README.md              # how to run everything
├── configs/
│   ├── cnn.yaml
│   ├── transformer.yaml
│   └── hybrid.yaml
├── src/
│   ├── data.py            # download check, loading, label mapping, splits, caching
│   ├── preprocess.py      # filtering + standardization
│   ├── dataset.py         # PyTorch Dataset / DataLoaders
│   ├── models/
│   │   ├── cnn.py         # 1D-CNN baseline
│   │   ├── transformer.py # patch Transformer (own encoder block so attention is accessible)
│   │   └── hybrid.py      # CNN front-end + Transformer
│   ├── train.py           # training loop, early stopping, checkpointing, logging
│   ├── evaluate.py        # metrics, threshold tuning, results tables
│   ├── explain.py         # attention maps / rollout + plots
│   ├── llm_report.py      # LLM report generation
│   └── utils.py           # seeds, device, timing
├── notebooks/
│   └── presentation.ipynb # loads saved results/figures; no training inside
├── results/               # metrics JSON/CSV, checkpoints, logs
├── figures/               # PNGs for slides
└── docs/
    ├── explanations.md    # my study notes (one entry per milestone)
    └── qa_prep.md         # likely professor questions + answers
```

Each training run is launched like: `python -m src.train --config configs/transformer.yaml`.

---

## 6. Milestones

### Milestone 1: Data loading + exploration
- Implement `data.py`: download check, loading at 100 Hz, label mapping, fold split, caching.
- Produce:
  - `figures/class_distribution.png`: count per superclass (train/val/test).
  - `figures/example_ecgs.png`: one example ECG per class (leads II and V2 are enough).
  - A printed summary: number of ECGs per split, label counts, % of multi-label ECGs.
- **Done when:** cached arrays exist, the shapes are `(N, 1000, 12)` and `(N, 5)`, and the figures look right.

### Milestone 2: Preprocessing + DataLoaders
- Implement `preprocess.py` and `dataset.py`.
- Produce `figures/preprocessing_before_after.png` (one lead, raw vs. filtered).
- Model input shape convention: CNN uses `(batch, 12, 1000)`; Transformer converts internally.
- **Done when:** a batch loads with correct shapes, and the standardization stats come from train only.

### Milestone 3: Training + evaluation framework + CNN baseline
- `train.py`:
  - Loss: `BCEWithLogitsLoss`. Try `pos_weight` (neg/pos ratio per class from train) as a config option for imbalance.
  - Optimizer: AdamW, weight decay 0.01, batch size 64.
  - Up to 50 epochs, **early stopping on validation macro AUROC** (patience 8). Save the best checkpoint.
  - Log per epoch: train loss, val loss, val macro AUROC, time. Save to `results/<run>/history.csv`.
- `evaluate.py`:
  - **Macro AUROC** and **per-class AUROC** (`sklearn.metrics.roc_auc_score`).
  - **Threshold tuning:** for each class, pick the threshold that maximizes F1 on **validation**, then apply it to test.
  - Per-class **F1**, precision, recall on test.
  - Parameter count and total training time.
  - Save `results/<run>/metrics.json` and append a row to `results/summary.csv`.
- CNN baseline (`models/cnn.py`): 4 blocks of Conv1d(kernel 7) → BatchNorm → ReLU → MaxPool(2), channels 32→64→128→128, global average pooling, dropout 0.3, Linear → 5 logits. lr 1e-3.
- Produce `figures/training_curves_cnn.png`.
- **Done when:** the CNN trains end-to-end and test macro AUROC is reported (expect roughly 0.85–0.92).

### Milestone 4: Patch Transformer (main model)
- `models/transformer.py`:
  - Input `(batch, 1000, 12)`. Split time into **20 non-overlapping patches of 50 steps (0.5 s)**. Each patch covers all 12 leads → flatten to 600 values.
  - Linear patch embedding 600 → `d_model = 128`, prepend a learnable **[CLS] token**, add **learnable positional embeddings** (21 positions).
  - **Write a simple encoder block yourself** (LayerNorm → multi-head self-attention → residual → LayerNorm → MLP → residual) using `nn.MultiheadAttention(..., batch_first=True)` and return the attention weights (`need_weights=True, average_attn_weights=False`). This is needed for explainability.
  - 4 layers, 4 heads, MLP size 256, dropout 0.1.
  - Classification head on the [CLS] output → 5 logits.
- Training: AdamW lr 3e-4, weight decay 0.05, linear warmup (5 epochs) then cosine decay, gradient clipping at 1.0. Otherwise same as Milestone 3.
- Produce `figures/training_curves_transformer.png`.
- **Done when:** the Transformer is trained and compared to the CNN in `results/summary.csv`.
- If the Transformer is clearly worse than the CNN, tell me, then try one quick variant (e.g. patch size 25, or 2 layers) before moving to Milestone 5. Don't spend long on tuning.

### Milestone 5: Hybrid CNN + Transformer (only if needed)
- `models/hybrid.py`: 2–3 Conv1d blocks (downsample time to ~50–125 steps) → each time step becomes a token → the same Transformer encoder → [CLS] head.
- Train with the Transformer settings. Compare all models in `results/summary.csv`.
- **Done when:** I have a final comparison: CNN vs. Transformer vs. Hybrid.

### Milestone 6: Explainability
- `explain.py`:
  - **Attention rollout** from [CLS] to the 20 patches (multiply the head-averaged attention matrices across layers, adding identity for residuals).
  - Plot an ECG (lead II + one chest lead) with the patch importance shown as a shaded overlay along time.
  - Do this for 2 correctly predicted test examples each for MI, STTC and CD, plus 1–2 misclassified examples.
- Produce:
  - `figures/attention_<class>_<id>.png`
  - `figures/roc_curves.png` (per class, best model)
  - `figures/model_comparison.png` (macro AUROC bar chart, all models)
- In the milestone note, tell me whether the highlighted regions make medical sense. For example, for MI, does it focus around the ST segment (just after the QRS spike)? Be honest if it doesn't.

### Milestone 7: LLM report
- `llm_report.py`:
  - Input per ECG: predicted classes, probabilities, tuned thresholds, top 3 attended time windows (e.g. "3.5–4.0 s"), patient age and sex.
  - Read the API key from an environment variable (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; the provider is chosen in config). Never hard-code keys.
  - **If no key is set, fall back to a template-based report** so the pipeline still runs.
  - Prompt rules: plain language, 3–5 sentences, mention the confidence, name which time segments influenced the result, and end with "This is an AI research prototype, not a medical diagnosis." No made-up clinical details beyond the inputs.
  - Save 3–5 example reports to `results/llm_reports.md`.
- **Done when:** example reports exist and read sensibly.

### Milestone 8 (optional bonus): Autoencoder anomaly detection
- Only if I say so after Milestone 7.
- 1D convolutional autoencoder trained on **NORM-only** training ECGs. Anomaly score = reconstruction MSE.
- Evaluate normal vs. abnormal on test: AUROC. Produce a histogram of scores (normal vs. abnormal) and one example of a good vs. poor reconstruction.

### Milestone 9: Presentation assets + Q&A prep
- `notebooks/presentation.ipynb`: loads saved results and figures (no training). Includes: dataset summary, pipeline diagram (a simple matplotlib or text diagram is fine), results table, per-class results, attention figures, LLM report examples, limitations.
- `results/final_table.md`: a clean markdown table of all models with macro AUROC, per-class AUROC, macro F1, parameters and training time.
- `docs/qa_prep.md`: 15 likely professor questions with short answers based on *my actual results*, e.g.:
  - Why a Transformer for ECG? Why patches?
  - Why AUROC instead of accuracy?
  - How did you handle class imbalance?
  - Why the official folds? What is data leakage?
  - Why did the Transformer beat or lose to the CNN?
  - What does the attention map show, and can we trust it?
  - Which class is hardest and why?
  - What are the limitations and what would you do next?
- `README.md`: setup, how to run each step, where the outputs are.

---

## 7. Reference numbers (for sanity checks)

- Published deep models on PTB-XL superdiagnostic (5 classes) reach about **0.93 macro AUROC**.
- A reasonable target for this project: **≥ 0.88** for the best model. Below 0.80 probably means a bug: check the label mapping, the split, the normalization, and the input shape order.

---

## 8. Do not

- Do not use fold 10 for anything except the final evaluation.
- Do not compute normalization stats on val or test data.
- Do not shuffle patients across folds or make a random split.
- Do not hard-code API keys or commit data/checkpoints (add `data/`, `results/**/*.pt` to `.gitignore`).
- Do not add frameworks I didn't ask for (no Lightning, Hydra, W&B) unless I agree. Keep it plain PyTorch.
- Do not run everything in one go. Stop after each milestone.
