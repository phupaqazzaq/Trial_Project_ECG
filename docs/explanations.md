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
