# Study notes: one entry per milestone

Each entry has: what was built, what to look at, and how to explain it to the professor.

---

## Milestone 1: Data loading + exploration

**Built:** `src/data.py`, `src/utils.py`

**What it does**
1. Reads `ptbxl_database.csv` (one row per ECG) and `scp_statements.csv` (the dictionary of diagnosis codes).
2. For every ECG, looks at its `scp_codes`, keeps only the *diagnostic* codes, and maps each one to its superclass (NORM, MI, STTC, CD, HYP). The result is a set of 1 to 4 superclasses per ECG.
3. Drops the 411 ECGs that have no diagnostic superclass (21,799 → **21,388**).
4. Turns the labels into a multi-hot vector of length 5, e.g. an ECG with MI and CD is `[0, 1, 0, 1, 0]`.
5. Loads the 100 Hz waveforms with `wfdb.rdsamp()` → array `(21388, 1000, 12)` and caches everything in `data/cache/*.npy`.
6. Splits by the official `strat_fold` column: folds 1–8 train, 9 validation, 10 test.

**Results to look at**

| split | ECGs | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|
| train | 17,084 | 7,596 | 4,379 | 4,186 | 3,907 | 2,119 |
| val | 2,146 | 955 | 540 | 528 | 495 | 268 |
| test | 2,158 | 963 | 550 | 521 | 496 | 262 |

- Totals (NORM 9,514 / MI 5,469 / STTC 5,235 / CD 4,898 / HYP 2,649) are **identical to the numbers in the PTB-XL paper**, which confirms the label mapping is correct.
- **24.1 %** of ECGs have 2 or more superclasses (16,244 have one label, 4,068 two, 919 three, 157 four).
- Figures: `figures/class_distribution.png`, `figures/example_ecgs.png`. Table: `results/class_counts.csv`.

**How to explain this to my professor**
> PTB-XL has 21,799 ten-second, 12-lead ECGs. Each ECG comes with detailed diagnosis codes, and I map those codes to 5 broad superclasses using the official lookup table; ECGs without any diagnostic code are dropped, leaving 21,388. It is a *multi-label* problem because one patient can have several conditions at once (about 24 % do), so the target is a vector of five 0/1 values rather than a single class. The classes are imbalanced: normal is the most common and hypertrophy the rarest (about 12 %). I use the dataset's official folds (1–8 train, 9 validation, 10 test) because they keep every patient inside one fold, so the model is never tested on a patient it has already seen, and my numbers are comparable to published papers.

---

## Milestone 2: Preprocessing + DataLoaders

**Built:** `src/preprocess.py`, `src/dataset.py`

**What it does**
1. **Bandpass filter 0.5–40 Hz** (Butterworth, order 3, applied with `filtfilt`). Below 0.5 Hz is baseline drift (breathing, electrode movement); above 40 Hz is muscle noise and 50 Hz power-line interference. `filtfilt` runs the filter forwards and then backwards, so the peaks are not shifted in time.
2. **Standardization per lead**: `(x - mean) / std`, where mean and std (12 values each) are computed on the **training folds only** and saved to `data/cache/norm_stats.npz`. The same numbers are then applied to val and test.
3. `ECGDataset` returns each ECG as a tensor `(12, 1000)` = (leads, time) plus its label `(5,)`. The DataLoader makes batches of 64 → `(64, 12, 1000)`.

**Results to look at**
- `figures/preprocessing_before_after.png`: the raw signal slides from 0 to −1.5 mV (drift); after filtering it sits flat around 0 and the heartbeats are unchanged.
- Batch shape check: `x (64, 12, 1000)`, `y (64, 5)`.
- After standardization the train set has mean 0.000 and std 1.000 per lead. Val (std 0.958–1.012) and test (0.981–1.022) are close to 1 but not exactly 1. That is expected and is the proof that their statistics were *not* used.
- Chest leads have larger amplitude (std up to 0.31 mV for V2) than limb leads (about 0.12–0.14 mV), which is why each lead is standardized separately.

**How to explain this to my professor**
> Raw ECGs contain slow baseline wander and high-frequency noise that have nothing to do with the heart, so I apply a standard 0.5–40 Hz bandpass filter, which keeps the P wave, QRS complex and T wave. Then I standardize each lead to zero mean and unit variance, because the leads have different amplitudes and neural networks train better when inputs are on a similar scale. The important detail is that the mean and standard deviation are computed from the training set only and then reused for validation and test; computing them on all the data would leak information about the test set into training. I do not use data augmentation in this first version.

---

## Milestone 3: Training + evaluation framework + CNN baseline

**Built:** `src/train.py`, `src/evaluate.py`, `src/models/cnn.py`, `src/models/__init__.py`, `configs/cnn.yaml`, `configs/cnn_posweight.yaml`

**What it does**
- **Model (190,757 parameters):** 4 blocks of Conv1d(kernel 7) → BatchNorm → ReLU → MaxPool(2), channels 12→32→64→128→128. The time length shrinks 1000→500→250→125→62. Then global average pooling over time, dropout 0.3, and one Linear layer giving 5 logits.
- **Loss:** `BCEWithLogitsLoss`, which is a sigmoid plus binary cross-entropy *per class*. Each class is its own yes/no question, which is what a multi-label problem needs (softmax would force exactly one class).
- **Training:** AdamW (lr 1e-3, weight decay 0.01), batch size 64, up to 50 epochs. After every epoch the validation macro AUROC is computed; the best checkpoint is saved, and training stops when there is no improvement for 8 epochs (early stopping). Mixed precision is used on the CUDA GPU.
- **Evaluation:** macro AUROC and per-class AUROC. Then, for each class, the probability threshold that gives the best F1 **on validation** is found and applied unchanged to the test fold to get F1 / precision / recall.
- `--no-test` flag: experiments report validation numbers only, so the test fold is never used to choose between variants.

**Results to look at** (`results/cnn/metrics.json`, `figures/training_curves_cnn.png`)

- Best epoch 12 of 20, training time 30 s on an RTX 4050.
- Validation macro AUROC 0.9327, **test macro AUROC 0.9272**, test macro F1 0.749.

| class | test AUROC | F1 | precision | recall | threshold |
|---|---|---|---|---|---|
| NORM | 0.946 | 0.863 | 0.803 | 0.933 | 0.43 |
| MI | 0.923 | 0.750 | 0.723 | 0.778 | 0.35 |
| STTC | 0.939 | 0.770 | 0.714 | 0.835 | 0.25 |
| CD | 0.922 | 0.758 | 0.752 | 0.764 | 0.56 |
| HYP | 0.906 | 0.602 | 0.601 | 0.603 | 0.17 |

- **pos_weight experiment (class imbalance):** the weights were NORM 1.25, MI 2.90, STTC 3.08, CD 3.37, HYP 7.06. Validation macro AUROC was 0.9307 with pos_weight vs. 0.9327 without, so it did **not** help and is left off. Reason: AUROC only depends on how the model *ranks* ECGs, not on the threshold, and the imbalance is already handled at decision time by tuning a separate threshold per class (note HYP's low threshold of 0.17).
- The curves show mild overfitting after epoch 12: train loss keeps falling while val loss rises. Early stopping keeps the epoch-12 weights.
- 0.927 is slightly above the 0.85–0.92 range I expected and close to the ~0.93 of published deep models, so there is no sign of a bug in labels, split or normalization.

**How to explain this to my professor**
> The baseline is a small 1D convolutional network: four convolution blocks that slide filters along time to detect local wave shapes, followed by averaging over time and a linear layer with five outputs. Because one ECG can have several diagnoses, each output goes through its own sigmoid and the loss is binary cross-entropy per class. I stop training when the validation macro AUROC stops improving and keep the best checkpoint, which prevents overfitting. To turn probabilities into yes/no decisions I choose one threshold per class that maximizes F1 on the validation fold, then apply it to the test fold, so nothing is ever tuned on test data. The CNN reaches 0.927 macro AUROC on the test fold with only 190 k parameters and 30 seconds of training.

---

## Milestone 4: Patch Transformer (main model)

**Built:** `src/models/transformer.py`, `configs/transformer.yaml`, `configs/transformer_patch25.yaml`

**What it does**
1. The ECG `(1000 time steps, 12 leads)` is cut into **20 non-overlapping patches of 50 steps (0.5 s)**. Each patch holds all 12 leads and is flattened to 600 numbers.
2. A Linear layer maps each patch from 600 → 128 numbers (the "patch embedding"). Now the ECG is a sequence of 20 tokens.
3. A learnable **[CLS] token** is put in front (21 tokens), and a learnable **positional embedding** is added so the model knows the order of the patches.
4. **4 encoder blocks**, written by hand: `LayerNorm → multi-head self-attention (4 heads) → residual → LayerNorm → MLP (128→256→128) → residual`. In self-attention every token looks at all other tokens and decides how much to take from each. The block returns the attention weights, which Milestone 6 needs.
5. The final [CLS] vector goes through a Linear layer → 5 logits.
- Training: AdamW lr 3e-4, weight decay 0.05, 5 epochs linear warmup then cosine decay, gradient clipping at 1.0, dropout 0.1. 610,565 parameters.

**Results to look at** (`results/transformer/metrics.json`, `figures/training_curves_transformer.png`)

- Best epoch 11 of 19, training time 50 s.
- Validation macro AUROC 0.9003, **test macro AUROC 0.8901**, test macro F1 0.687.
- Per-class test AUROC: NORM 0.918, MI 0.857, STTC 0.915, CD 0.880, HYP 0.881.
- **The Transformer is clearly worse than the CNN** (0.890 vs. 0.927 on test; 0.900 vs. 0.933 on validation), although it has 3× more parameters. The biggest drops are MI (−0.066) and CD (−0.042).
- It overfits quickly: from epoch 6 the train loss keeps falling (0.30 → 0.19) while the validation loss rises (0.32 → 0.37).
- **Quick variant (as planned):** patch size 25 (40 patches of 0.25 s) gave validation macro AUROC 0.9051 vs. 0.9003, a difference too small to matter for a single run. It did not close the gap, so I stopped tuning and moved to the hybrid model. The variant was judged on validation only and never run on the test fold.

**How to explain this to my professor**
> The Transformer treats the ECG like a sentence: I cut the 10-second signal into 20 half-second patches, turn each patch into a vector, and let self-attention relate every patch to every other patch, so the model can compare beats that are far apart in time. A special [CLS] token collects the information and is used for the final prediction. It reached 0.890 test macro AUROC, which meets the 0.88 target but is clearly below the CNN's 0.927. The reason is that a pure Transformer has no built-in knowledge about signals: the linear patch embedding must learn from scratch what a QRS spike looks like, and a patch boundary can fall in the middle of a heartbeat, so the same beat looks different depending on where it falls. A CNN has this "local shape, any position" assumption built in. With only 17,000 training ECGs the Transformer overfits, which matches the well-known result that Transformers need much more data (or pre-training) than CNNs.

---

## Milestone 5: Hybrid CNN + Transformer

**Built:** `src/models/hybrid.py`, `configs/hybrid.yaml`

**What it does**
- A CNN front-end made of the **same conv blocks as the baseline** (3 of them, channels 12→32→64→128) shrinks time 1000 → 125. Each of the 125 time steps (0.08 s each) becomes one token of size 128.
- The **same Transformer encoder** as Milestone 4 ([CLS] token, positional embeddings, 4 layers, 4 heads) processes the 125 tokens, and the [CLS] output gives the 5 logits.
- Same training settings as the Transformer. 622,117 parameters.

**Results to look at** (`results/summary.csv`, `results/hybrid/metrics.json`, `figures/training_curves_hybrid.png`)

| model | parameters | train time | val macro AUROC | **test macro AUROC** | test macro F1 |
|---|---|---|---|---|---|
| CNN | 190,757 | 30 s | 0.9327 | **0.9272** | 0.749 |
| Patch Transformer | 610,565 | 50 s | 0.9003 | **0.8901** | 0.687 |
| Hybrid CNN + Transformer | 622,117 | 99 s | 0.9277 | **0.9235** | 0.735 |

- Hybrid per-class test AUROC: NORM 0.945, MI 0.919, STTC 0.931, CD 0.920, HYP 0.902.
- The hybrid recovers almost the whole gap: +0.033 over the pure Transformer, and only 0.004 below the CNN. I trained each model once (seed 42), and a 0.004 difference on a 2,158-ECG test fold is too small to call one of them better.
- **HYP is the hardest class for every model** (AUROC 0.88–0.91, F1 0.56–0.60): it is the rarest class and usually appears together with other labels.

**How to explain this to my professor**
> Since the pure Transformer lost to the CNN, I combined them: a small CNN first extracts local wave shapes and reduces the signal to 125 time steps, and then the Transformer encoder uses attention across those 125 tokens. This fixes the weak point of the pure Transformer, because the convolutions provide the local pattern detection that the linear patch embedding could not learn from 17,000 examples. The hybrid reaches 0.924 test macro AUROC, essentially the same as the CNN (0.927) and far better than the pure Transformer (0.890). So my conclusion is that on a dataset of this size the convolutional front-end is what matters most; attention on top neither helps nor hurts the accuracy, but it does give attention maps for explainability, which the plain CNN cannot provide.

---

## Milestone 6: Explainability

**Built:** `src/explain.py`

**What it does**
- **Attention rollout.** Each of the 4 layers gives an attention matrix (21 × 21 for the patch Transformer: [CLS] + 20 patches). For one ECG: (1) average the 4 heads, (2) add the identity matrix and re-normalize, because the residual connection lets every token also keep its own information, (3) multiply the matrices of all layers, (4) read the [CLS] row. The result is one number per patch: how much that patch contributed to the [CLS] output, summing to 100 %.
- **Plots:** lead II and V2 with the importance as red shading, plus a bar chart of the attention share per 0.5 s window. The dashed line at 5 % is what "equal attention to all 20 windows" would look like.
- Examples: the 2 most confident correct single-label test ECGs for MI, STTC and CD, plus one missed MI and one false MI alarm. Done for the patch Transformer (`attention_<class>_<id>.png`) and, as an extra, for the hybrid (`attention_hybrid_<class>_<id>.png`), whose tokens are much finer (0.08 s instead of 0.5 s).
- `attention_statistics`: the same analysis as numbers over all 2,158 test ECGs, so the conclusion does not depend on a few hand-picked plots.
- Also: `figures/roc_curves.png` (best model = CNN) and `figures/model_comparison.png`.

**Results to look at**

| | Patch Transformer | Hybrid |
|---|---|---|
| Share of attention on the strongest 0.5 s window (equal = 5 %) | 7.7 % | 6.9 % |
| Correlation between a window's attention and its signal energy (Spearman) | +0.19 | +0.50 |
| ECGs where that correlation is positive | 73 % | 89 % |
| Average attention by window position | flat, 4.8–5.1 % | flat, 4.9–5.2 % |

**Do the highlighted regions make medical sense? Honest answer: only partly.**

1. **Patch Transformer: the attention is diffuse and too coarse to interpret.** The strongest window gets only 7.7 % on average, barely above the 5 % of looking everywhere equally. A patch is 0.5 s and a heartbeat lasts about 0.8–1 s, so one patch contains a QRS complex together with its ST segment, or a T wave together with the flat baseline. At this resolution it is **impossible** to say that the model looks at the ST segment for MI. There is only a weak tendency (+0.19) to prefer patches that contain a heartbeat over flat ones.
2. **It sometimes highlights noise.** In `attention_MI_11238.png` the most attended window (9.5–10.0 s, 10 %) is a noise artifact at the end of the recording, not a cardiac feature, even though the MI prediction (0.99) is correct.
3. **Hybrid: clearer and beat-locked.** The attention forms narrow bands that repeat once per heartbeat, on and around the QRS complex (correlation +0.50, positive in 89 % of ECGs). The flat baseline between beats gets little attention.
   - **CD** (`attention_hybrid_CD_18671.png`): the bands sit on the wide, notched QRS complexes. This **does** make medical sense, because conduction disturbances such as bundle branch block are diagnosed from the width and shape of the QRS.
   - **MI** (`attention_hybrid_MI_6552.png`): the bands sit at the QRS complex in every beat, **not** on the ST segment after it. Abnormal Q waves inside the QRS are a real sign of an old infarction, so this is plausible, but I cannot claim that the model looks at ST elevation.
   - **STTC** (`attention_hybrid_STTC_17323.png`): bands just before and just after the QRS; some touch the start of the ST segment, but the T wave itself is not clearly highlighted.
4. **No position bias:** averaged over the test fold every window position gets about 5 %, so the model is not simply staring at the start or end of the recording.
5. **Misclassified examples** (`attention_misclassified_*.png`): the attention pattern of a wrong prediction looks the same as that of a correct one. So the map does not warn us when the model is wrong.

**Caveats I should state**
- Attention shows where information *flows*, not what *caused* the decision ("attention is not explanation"). Rollout also ignores the MLP layers.
- In the hybrid, each token comes out of 3 convolution layers and "sees" about 0.4 s of signal, so a 0.08 s band is not as precise as it looks.
- Only leads II and V2 are plotted, but the model uses all 12 leads; the attention is over time only and cannot say *which lead* mattered.

**How to explain this to my professor**
> I used attention rollout, which multiplies the attention matrices of all layers (adding the identity for the residual connections) to estimate how much each time patch contributes to the [CLS] token that makes the prediction. For the pure patch Transformer the result is honest but disappointing: attention is spread almost evenly, the strongest half-second window gets about 8 % versus 5 % for uniform, and a half-second patch is too coarse to separate the ST segment from the QRS complex. The hybrid model gives sharper maps that lock onto the QRS complex of every beat, which is medically sensible for conduction disturbances, since those are defined by QRS width and shape. For myocardial infarction it also highlights the QRS region rather than the ST segment, so I cannot claim it has learned ST elevation. I treat these maps as a sanity check that the model looks at heartbeats and not at noise or a fixed position, not as proof of clinical reasoning.

---

## Milestone 7: LLM report

**Built:** `src/llm_report.py`, `configs/llm.yaml`

**What it does**
1. For one test ECG it collects a small "case" dictionary: patient age and sex, the 5 probabilities, the 5 tuned thresholds, the predicted classes (probability ≥ threshold), and the top 3 attended 0.5 s windows from attention rollout (e.g. "3.5–4.0 s").
2. **If `ANTHROPIC_API_KEY` is set**, the case is sent as JSON to Claude (`claude-opus-5-5`, chosen in `configs/llm.yaml`) with a system prompt containing the rules: plain language, 3–5 sentences, mention the confidence, name the time segments, use only the given inputs, and end with "This is an AI research prototype, not a medical diagnosis."
3. **If no key is set (or the call fails)**, a template function fills the same information into fixed sentences, so the pipeline always runs.
4. Five examples (one correctly predicted test ECG per class) are saved to `results/llm_reports.md` together with the exact inputs each report was built from.
- The model used is the **hybrid**, because it is the best model that has attention (the CNN is slightly better but has no attention maps).

**Results to look at:** `results/llm_reports.md`

**Important: what was and was not tested**
- No API key was set on this machine, so **all 5 saved reports come from the template**, and the file says so ("source: template").
- The Claude code path is written following the current SDK documentation, but it **has never been run**. Before presenting, set `ANTHROPIC_API_KEY` and run `python -m src.llm_report` once to check it. Only `provider: anthropic` is implemented; an OpenAI branch was not written.

**How to explain this to my professor**
> The last step turns numbers into words. For each ECG I pass only structured facts to the language model: the predicted classes, their probabilities and thresholds, the three most attended time windows, and the patient's age and sex. The prompt tells it to write 3 to 5 plain sentences, state the confidence, name the time segments, add nothing that is not in the input, and finish with a disclaimer. The LLM never sees the ECG signal and does not diagnose anything; it only rephrases the classifier's output, which limits the risk of made-up medical details. If no API key is available the same facts go into a fixed template, so the pipeline still works. The examples currently in the repository were produced by that template.
