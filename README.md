# ECG Disease Prediction: Patch Transformer + LLM Report

Trial project (Natdanai, MUICT). Predicts 5 diagnostic superclasses from 12-lead ECGs (PTB-XL), compares a 1D-CNN, a patch Transformer and a hybrid CNN + Transformer, explains predictions with attention maps, and writes a short plain-language report.

## Results (official PTB-XL test fold)

| Model | Parameters | Train time | Macro AUROC | NORM | MI | STTC | CD | HYP | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| 1D-CNN (baseline) | 190,757 | 30 s | **0.9272** | 0.946 | 0.923 | 0.939 | 0.922 | 0.906 | 0.748 |
| Patch Transformer | 610,565 | 50 s | **0.8901** | 0.918 | 0.857 | 0.915 | 0.880 | 0.881 | 0.687 |
| Hybrid CNN + Transformer | 622,117 | 99 s | **0.9235** | 0.945 | 0.919 | 0.931 | 0.920 | 0.902 | 0.735 |

One run per model (seed 42) on an RTX 4050 laptop GPU. Published deep models reach about 0.93.

![Model comparison](figures/model_comparison.png)

**Bonus: anomaly detection.** A convolutional autoencoder (32,980 parameters) trained on normal ECGs only, with the reconstruction error as anomaly score, separates normal from abnormal test ECGs with AUROC **0.755**. For reference: signal energy alone gives 0.612, and the supervised CNN gives 0.939 on the same question.

![Anomaly scores](figures/anomaly_score_histogram.png)

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

# PyTorch: pick the command for your system from https://pytorch.org/get-started/locally/
# Example for an NVIDIA GPU (CUDA 12.6):
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install numpy pandas scipy scikit-learn wfdb matplotlib tqdm pyyaml anthropic jupyter
```

The device is chosen automatically: CUDA → MPS (Apple) → CPU. On Google Colab, PyTorch is already installed; run only the last `pip install` line.

## Data

PTB-XL v1.0.3 (open access): https://physionet.org/content/ptb-xl/1.0.3/

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/ptb-xl https://physionet.org/files/ptb-xl/1.0.3/
```

Or download the zip from the PhysioNet page and unzip it so that these exist:

```
data/ptb-xl/ptbxl_database.csv
data/ptb-xl/scp_statements.csv
data/ptb-xl/records100/
```

Only the 100 Hz records (`records100`) are needed. `data/` is not committed to git.

## How to run each step

Run everything from the project root.

| Step | Command | Main outputs |
|---|---|---|
| 1. Load data, build labels, cache | `python -m src.data` | `data/cache/*.npy`, `figures/class_distribution.png`, `figures/example_ecgs.png`, `results/class_counts.csv` |
| 2. Filter + normalization stats | `python -m src.preprocess` | `data/cache/X_filtered.npy`, `data/cache/norm_stats.npz`, `figures/preprocessing_before_after.png` |
| 2b. Check a batch | `python -m src.dataset` | prints batch shapes |
| 3. Train CNN baseline | `python -m src.train --config configs/cnn.yaml` | `results/cnn/`, `figures/training_curves_cnn.png` |
| 4. Train patch Transformer | `python -m src.train --config configs/transformer.yaml` | `results/transformer/` |
| 5. Train hybrid | `python -m src.train --config configs/hybrid.yaml` | `results/hybrid/` |
| 6. Attention maps, ROC, comparison | `python -m src.explain --run transformer --roc-run cnn --compare cnn transformer hybrid` | `figures/attention_*.png`, `figures/roc_curves.png`, `figures/model_comparison.png` |
| 6b. Attention maps for the hybrid | `python -m src.explain --run hybrid --prefix attention_hybrid` | `figures/attention_hybrid_*.png` |
| 7. Reports | `python -m src.llm_report --config configs/llm.yaml` | `results/llm_reports.md` |
| 8. Final table | `python -m src.evaluate --final-table cnn transformer hybrid` | `results/final_table.md` |
| 9. Bonus: autoencoder anomaly detection | `python -m src.anomaly --config configs/autoencoder.yaml` | `results/autoencoder/`, `figures/anomaly_score_histogram.png`, `figures/anomaly_reconstruction_examples.png` |
| 10. Presentation | open `notebooks/presentation.ipynb` | loads saved results, no training |

Notes:
- Each training run writes `results/<run>/`: `config.yaml`, `history.csv`, `best.pt` (not committed), `metrics.json`, and adds a row to `results/summary.csv`.
- Step 9 uses the CNN's saved test predictions for its reference number, so run it after step 3.
- Add `--no-test` to a training command for experiments. It reports validation numbers only, so the test fold (fold 10) is never used to choose between variants. The experiment configs `configs/cnn_posweight.yaml` and `configs/transformer_patch25.yaml` were run this way.
- All seeds are fixed to 42.

### LLM report and API key

Step 7 calls Claude only if the environment variable `ANTHROPIC_API_KEY` is set (never put the key in the code). Without it, a template-based report is produced, so the pipeline always runs. The example reports in this repository were produced by the template.

```bash
# Windows PowerShell:  $env:ANTHROPIC_API_KEY = "..."      macOS/Linux:  export ANTHROPIC_API_KEY=...
python -m src.llm_report --config configs/llm.yaml
```

## Project structure

```
configs/        one YAML file per training run + llm.yaml
src/
  data.py         loading, label mapping, fold split, caching, exploration figures
  preprocess.py   bandpass filter + standardization (train statistics only)
  dataset.py      PyTorch Dataset / DataLoaders
  models/         cnn.py, transformer.py (own encoder block), hybrid.py, autoencoder.py
  train.py        training loop, early stopping, checkpointing, logging
  evaluate.py     AUROC, threshold tuning on validation, F1, result tables
  explain.py      attention rollout, attention plots, ROC curves, comparison chart
  llm_report.py   plain-language report (Claude API or template fallback)
  anomaly.py      bonus: autoencoder trained on normal ECGs, anomaly scores, figures
  utils.py        seeds, device, parameter count
notebooks/      presentation.ipynb
results/        metrics, tables, example reports
figures/        PNGs for slides
docs/
  explanations.md   study notes, one entry per milestone
  qa_prep.md        15 likely questions with answers
```

## Disclaimer

This is an AI research prototype, not a medical diagnosis tool.
