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
