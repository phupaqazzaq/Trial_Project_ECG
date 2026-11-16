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
- Validation macro AUROC 0.9327, **test macro AUROC 0.9272**, test macro F1 0.748.

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
| CNN | 190,757 | 30 s | 0.9327 | **0.9272** | 0.748 |
| Patch Transformer | 610,565 | 50 s | 0.9003 | **0.8901** | 0.687 |
| Hybrid CNN + Transformer | 622,117 | 99 s | 0.9277 | **0.9235** | 0.735 |

- Hybrid per-class test AUROC: NORM 0.945, MI 0.919, STTC 0.931, CD 0.920, HYP 0.902.
- The hybrid recovers almost the whole gap: +0.033 over the pure Transformer, and only 0.004 below the CNN. I trained each model once (seed 42), and a 0.004 difference on a 2,158-ECG test fold is too small to call one of them better.
- **HYP is the hardest class for every model** (AUROC 0.88–0.91, F1 0.56–0.60): it is the rarest class, and 79.8 % of HYP ECGs also carry another label (MI 53.7 %, STTC 54.2 %, CD 65.1 %, NORM 4.7 %).

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

**Built:** `src/llm_report.py`, `configs/llm.yaml`, `src/app.py` (the free Qwen model, the automatic checks and the test website were added later, replacing the first version that only had Claude + template)

**What it does**
1. For one test ECG it collects a small "case" dictionary: patient age and sex, the 5 probabilities, the 5 tuned thresholds, the predicted classes (probability ≥ threshold), and the top 3 attended 0.5 s windows from attention rollout (e.g. "3.5–4.0 s").
2. The case is sent as JSON to a language model with a system prompt containing the rules: plain language, 3–5 sentences, mention the confidence, name the time segments, use only the given inputs, and end with "This is an AI research prototype, not a medical diagnosis." The prompt also contains one worked example. Which model is used is set by `provider` in `configs/llm.yaml`:
   - `qwen` (**the current setting**): **Qwen2.5-1.5B-Instruct**, a free open model that runs on my own laptop through the Hugging Face `transformers` library. No account, no API key, no internet needed after the one-time download (about 3 GB; about 3 GB of GPU memory; about 3 s per report). Decoding is greedy (`do_sample=False`), so the same input always gives the same report.
   - `anthropic`: Claude through the API, key read from the `ANTHROPIC_API_KEY` environment variable.
3. **Every LLM text is checked by code** (`check_report`) before it is accepted, because a language model can ignore its instructions:
   1. it ends with the disclaimer;
   2. it is one paragraph of 3 to 5 sentences;
   3. every number in it appears in the inputs (no invented values);
   4. it names every predicted class;
   5. it names at least one attended time window;
   6. its confidence words agree with the inputs (the words "high / moderate / low" come from fixed cut-offs: ≥ 0.8, ≥ 0.5, below);
   7. it contains no wording that signals interpretation or advice (a short word list such as "suggest", "should", "severe", "heart attack", "abnormal").
4. **If the LLM is not available or its text fails a check**, a template function fills the same information into fixed sentences, so the pipeline always runs. The source of every report ("LLM … passed the automatic checks" or "template (…reason…)") is saved next to it.
5. Five examples (one correctly predicted test ECG per class) are saved to `results/llm_reports.md` together with the exact inputs each report was built from.
- The model used is the **hybrid**, because it is the best model that has attention (the CNN is slightly better but has no attention maps).

**Results to look at:** `results/llm_reports.md`, `results/llm_check.json`

- **4 of the 5 saved example reports are written by Qwen.** The fifth (ECG 162, four predicted classes) was rejected because Qwen answered with a bullet list, so the template was used; the file says so.
- **Acceptance rate:** on 100 random validation ECGs, **91 Qwen reports passed all checks** and 9 fell back to the template (`python -m src.llm_report --check-val 100`). Reasons for the 9: too many sentences or a list (5), a number that was not in the inputs (2), "high" confidence claimed for a moderate prediction (2).

**Why the checks are needed (what I saw while building this)**
- A 1.5-billion-parameter model is small. With only the rules in the prompt, it wrote things like "indicates a severe heart attack", "suggesting possible arrhythmia" and "further investigation might be necessary", and for an ECG with no predicted class it wrote "classified as normal". None of that is in the inputs.
- Three things fixed most of it: (1) the decision is spelled out in the input as a sentence (`model_decision`, including the confidence word), so the model does not have to work it out from numbers; (2) one worked example in the prompt; (3) the automatic checks, which reject what still goes wrong.
- **The prompt and the checks were developed on validation ECGs.** Two checks ("one paragraph" and "confidence words") were added after I saw two flawed reports among the five test examples. This does not affect any classification metric.

**Honest limits**
- The checks catch broken *rules*, not every wrong *sentence*. An accepted report can still be clumsy (e.g. "MI has high confidence due to probabilities above the decision threshold"), and a false statement that uses no numbers and no listed words would pass.
- The word list in check 7 is crude: it can reject harmless text, and it cannot list every possible interpretation.
- The Claude path (`provider: anthropic`) **has still never been run**, because no API key is set on this machine. An OpenAI branch was not written.

**Test website** (`src/app.py`, run with `python -m src.app`)
- A small local web page built with Gradio. It runs on my laptop only (http://127.0.0.1:7860); nothing is uploaded.
- Pick one of the 2,158 test ECGs (or "Random ECG") and it shows: the probability table with thresholds and true labels, the attention figure, the report with its source, and the exact JSON the report generator received.
- It contains no new method. It calls the same functions as the scripts (`explain_ecg`, `draw_attention`, `build_case`, `generate_report`).

**How to explain this to my professor**
> The last step turns numbers into words. For each ECG I pass only structured facts to a language model: the predicted classes with their probabilities, thresholds and confidence level, the three most attended time windows, and the patient's age and sex. The model is Qwen2.5-1.5B, a free open model that runs locally on my laptop, so no patient data leaves the machine and no API key is needed. It never sees the ECG signal and does not diagnose anything; it only rephrases the classifier's output. Because a small language model does not always follow instructions, every text is checked by code: it must end with the disclaimer, have 3 to 5 sentences, contain only numbers that are in the inputs, name the predicted classes and a time window, use the right confidence words, and avoid interpretation words. On 100 validation ECGs, 91 reports passed; the other 9 were replaced by a fixed template. So the report is fluent when the LLM behaves and still correct when it does not.

---

## Milestone 8: Autoencoder anomaly detection (optional bonus)

**Built:** `src/models/autoencoder.py`, `src/anomaly.py`, `configs/autoencoder.yaml` (done after Milestone 9, when I asked for it)

**What it does**
1. **A different question.** The classifiers answer "which of the 5 classes?" using labels. The autoencoder answers "does this ECG look unusual?" and is trained **without any disease labels**: it only ever sees normal ECGs.
2. **Normal vs. abnormal.** Normal = NORM is the *only* label. Abnormal = any ECG with MI, STTC, CD or HYP (including the few that have NORM plus another label). Train: 7,243 normal ECGs of 17,084. Test: 912 normal, 1,246 abnormal.
3. **The model** (`ConvAutoencoder`, 32,980 parameters):
   - Encoder: 3 convolutions with stride 2, so the time axis shrinks 1000 → 500 → 250 → 125, ending with 8 channels. This **bottleneck** holds 125 × 8 = 1,000 numbers for an ECG of 12 × 1000 = 12,000 numbers (12× compression).
   - Decoder: the mirror image with transposed convolutions, 125 → 250 → 500 → 1000, back to 12 leads.
   - Because of the bottleneck the model cannot copy its input. It has to learn what normal heartbeats look like in order to rebuild them from so few numbers.
4. **Training:** loss = mean squared error (MSE) between the input and its reconstruction, on the normal training ECGs only. AdamW, lr 1e-3, batch 64, seed 42. Early stopping on the reconstruction loss of the **normal validation ECGs** (patience 8), so no abnormal label is used anywhere in training. Same preprocessing as the classifiers (bandpass filter, standardization with train statistics).
5. **Anomaly score** = reconstruction MSE of one ECG, averaged over all leads and time steps. High score = "the model could not rebuild this well" = unusual.
6. **Evaluation:** AUROC of the score for normal vs. abnormal. No threshold is needed for AUROC.

**Results to look at:** `results/autoencoder/metrics.json`, `figures/anomaly_score_histogram.png`, `figures/anomaly_reconstruction_examples.png`, `figures/training_curves_autoencoder.png`

| Normal vs. … | Test AUROC |
|---|---|
| **all abnormal ECGs** | **0.755** (validation: 0.779) |
| MI | 0.773 |
| STTC | 0.750 |
| CD | 0.816 |
| HYP | 0.873 |
| *Reference: signal energy alone, no model* | *0.612* |
| *Reference: supervised CNN, score = 1 − P(NORM)* | *0.939* |

- Training ran the full 50 epochs (best epoch 46, 21 s). Validation loss fell from 0.094 to 0.025, so the model rebuilds normal ECGs closely (the leads are standardized to a variance of about 1, so an MSE of 0.025 is a small error).
- Median score: 0.020 for normal test ECGs, 0.040 for abnormal ones. The histogram shows the abnormal scores shifted to the right, but **the two groups overlap a lot**.

**Honest reading of the result**
- **It works, but only moderately.** 0.755 is clearly better than random (0.5) and better than the trivial "signal energy" score (0.612), so the autoencoder has learned something about normal ECG *shape*, not only amplitude. But it is far below the supervised CNN (0.939) on the same question. That is expected: the CNN was shown thousands of labeled abnormal examples, the autoencoder none.
- **Why the overlap?** The autoencoder generalizes too well. Abnormal ECGs are still made of P waves, QRS complexes and T waves, so the model rebuilds most of them almost as well as normal ones. In the "poor reconstruction" example (ECG 18548, CD, score 0.292) the red line still follows the signal; the error comes mainly from the sharp peaks.
- **The class ranking is the opposite of the classifiers.** HYP is the *easiest* to detect here (0.873) but the hardest to classify. My explanation: hypertrophy produces large voltages, and large or wide deflections give a larger squared error. CD (0.816) changes the QRS shape strongly. STTC (0.750) is the hardest: ST/T changes are small, smooth shape changes that are easy to rebuild.
- **The score also reacts to noise.** The single highest score on the test fold belongs to a *normal* ECG (ID 11124, score 4.28): lead V6 has an electrode artifact reaching about 13 mV (a normal peak is about 1 mV). The detector finds "unusual recordings", which is not the same as "disease".
- **No tuning was done.** One configuration, chosen in advance; the bottleneck size was not tuned. A first run with `--no-test` checked the validation AUROC (0.779), then the final run evaluated the test fold. The script was run once more on the unchanged configuration only to add the two reference numbers; all model numbers came out identical.

**How to explain this to my professor**
> As a bonus I tried anomaly detection, which needs no disease labels. A small convolutional autoencoder compresses each ECG to one twelfth of its size and rebuilds it, and it is trained on normal ECGs only. The idea is that it learns what a normal heartbeat looks like, so an abnormal ECG should be rebuilt worse; the reconstruction error is the anomaly score. On the test fold this separates normal from abnormal with an AUROC of 0.755. That is better than chance and better than just measuring signal amplitude (0.61), but much weaker than my supervised CNN on the same question (0.94). The reason is that the autoencoder also rebuilds abnormal ECGs fairly well, because they are made of the same basic waves, and that its error reacts to amplitude and noise as well as to disease. So it is useful as a label-free "this recording is unusual" flag, not as a replacement for the classifier.

---

## Milestone 9: Presentation assets + Q&A prep

**Built:** `notebooks/presentation.ipynb`, `results/final_table.md`, `docs/qa_prep.md`, `README.md`, and `write_final_table()` in `src/evaluate.py`

**What it does**
- `notebooks/presentation.ipynb` loads saved results and figures only (no training): dataset summary, pipeline diagram, preprocessing, results table, per-class results, training curves, attention figures, report examples, limitations. It is saved already executed, so the tables and charts are visible without running it.
- `results/final_table.md` is the clean table of the three final models (created by `python -m src.evaluate --final-table cnn transformer hybrid`).
- `docs/qa_prep.md` has 15 likely questions with answers based on my real numbers, plus a "quick facts" list.
- `README.md` explains setup, the command for every step, and where the outputs are.

**Results to look at**

| Model | Parameters | Train time | Test macro AUROC | NORM | MI | STTC | CD | HYP | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| 1D-CNN (baseline) | 190,757 | 30 s | **0.9272** | 0.946 | 0.923 | 0.939 | 0.922 | 0.906 | 0.748 |
| Patch Transformer | 610,565 | 50 s | **0.8901** | 0.918 | 0.857 | 0.915 | 0.880 | 0.881 | 0.687 |
| Hybrid CNN + Transformer | 622,117 | 99 s | **0.9235** | 0.945 | 0.919 | 0.931 | 0.920 | 0.902 | 0.735 |

**How to explain this to my professor (the whole project in one paragraph)**
> I built a full pipeline on PTB-XL, from raw 12-lead ECGs to a plain-language report. Using the official patient-wise folds and a multi-label setup with five diagnostic superclasses, a small 1D-CNN reaches 0.927 macro AUROC on the test fold, close to published results. My main model, a patch Transformer, reaches 0.890: it meets my 0.88 target but loses to the CNN, because it lacks the built-in local pattern detection of convolutions and overfits on 17,000 ECGs. A hybrid that puts a CNN in front of the Transformer recovers the gap (0.924). Attention rollout shows that the hybrid concentrates on the QRS complexes, which is sensible for conduction disturbances, but the maps are not strong enough to claim clinical reasoning. The main limitations are single runs without confidence intervals, one dataset, and attention being only a rough explanation.
