# Q&A preparation: 15 likely questions (+ 2 on the bonus)

All numbers come from my actual runs (`results/final_table.md`, `results/summary.csv`). Test numbers are on the official PTB-XL test fold (fold 10, 2,158 ECGs).

| Model | Parameters | Test macro AUROC | Test macro F1 |
|---|---|---|---|
| 1D-CNN (baseline) | 190,757 | **0.927** | 0.748 |
| Patch Transformer | 610,565 | **0.890** | 0.687 |
| Hybrid CNN + Transformer | 622,117 | **0.924** | 0.735 |

---

### 1. What is the task and what data did you use?
Predicting heart conditions from 12-lead ECGs. I use PTB-XL v1.0.3: 21,799 ten-second ECGs, in the 100 Hz version, so each ECG is an array of 1000 time steps × 12 leads. The targets are 5 diagnostic superclasses: NORM, MI, STTC, CD and HYP. It is a **multi-label** problem: 24.1 % of ECGs have two or more labels. After dropping 411 ECGs without a diagnostic label, 21,388 remain (17,084 train / 2,146 validation / 2,158 test).

### 2. Why a Transformer for ECG? Why patches?
An ECG is a sequence, and a diagnosis often depends on relations between distant parts: rhythm across beats, or whether every beat shows the same abnormality. Self-attention connects any two time positions in a single step, while a CNN only sees a local window per layer. Patches are needed because attention cost grows with the square of the sequence length: 1000 single time steps would be expensive and each step alone carries almost no meaning. Cutting the signal into 20 patches of 0.5 s gives 20 meaningful tokens, the same idea as the Vision Transformer uses for images.

### 3. Why AUROC instead of accuracy?
The classes are imbalanced: HYP is in only about 12 % of ECGs, so a model that always answers "no HYP" would be 88 % accurate and useless. AUROC measures how well the model *ranks* positive cases above negative ones, across all possible thresholds, so it does not depend on the class balance or on one chosen threshold. 0.5 means random and 1.0 is perfect. I report **macro** AUROC (the plain average over the 5 classes) so the rare classes count as much as the common ones. It is also the standard metric in PTB-XL papers, so my numbers are comparable.

### 4. How did you handle class imbalance?
Three ways. (1) The metric: macro AUROC treats every class equally. (2) Decision thresholds: instead of 0.5 for everything, I tune one threshold per class to maximize F1 on the validation fold; for the CNN, HYP gets 0.17 while CD gets 0.56. (3) I tested `pos_weight` in the loss (weights from 1.25 for NORM to 7.06 for HYP). It did not help: validation macro AUROC was 0.9307 with it and 0.9327 without, so I left it off. That is expected, because pos_weight mostly shifts the probabilities, which AUROC ignores and the tuned thresholds already correct.

### 5. Why the official folds? What is data leakage?
Data leakage means information from the test data reaches training, which makes results look better than they really are. In PTB-XL some patients have several ECGs; with a random split, the same patient could be in both train and test, and the model could partly recognize the *patient* instead of the disease. The official `strat_fold` column keeps each patient in one fold and keeps the label proportions similar. I use folds 1–8 for training, 9 for validation and 10 for testing, which is also what published papers do. Other leakage I avoided: normalization statistics come from the training folds only, and thresholds and model choices are made on validation only.

### 6. How exactly did you use the test set?
Once per final model, at the end. Early stopping, threshold tuning, the pos_weight decision, and the patch-size variant were all judged on the validation fold. My training script has a `--no-test` flag for experiments, so the two experimental runs (`cnn_posweight`, `transformer_patch25`) have no test numbers at all in `results/summary.csv`.

### 7. What preprocessing did you do and why?
A 0.5–40 Hz Butterworth bandpass filter (order 3, applied forwards and backwards with `filtfilt` so peaks are not shifted). Below 0.5 Hz is baseline drift from breathing and electrode movement; above 40 Hz is muscle noise and 50 Hz mains interference. Then each lead is standardized to zero mean and unit variance using training-set statistics, because the leads have different amplitudes (std 0.12 mV for aVF vs. 0.31 mV for V2) and networks train better with inputs on the same scale. No augmentation was used.

### 8. Explain the Transformer architecture you built.
The input (1000 × 12) is cut into 20 patches of 50 time steps; each patch with all 12 leads is flattened to 600 numbers and a linear layer maps it to a 128-dimensional token. A learnable [CLS] token is added in front and learnable positional embeddings are added so the model knows the patch order. Then come 4 encoder blocks that I wrote myself: LayerNorm → multi-head self-attention (4 heads) → residual, then LayerNorm → MLP (128→256→128) → residual. The output at the [CLS] position goes through a linear layer to 5 logits. It has 610,565 parameters. I wrote the block by hand so that it returns the attention weights for the explainability step.

### 9. Why did the Transformer lose to the CNN?
The CNN scored 0.927 and the Transformer 0.890. A convolution has two built-in assumptions that fit ECGs: patterns are local (a QRS spike is about 0.1 s wide) and they look the same wherever they occur in time. The Transformer has neither: its linear patch embedding must learn every wave shape from scratch, and because the patch grid is fixed, the same heartbeat looks different depending on where it falls relative to the patch boundaries. With only 17,084 training ECGs it overfits: from epoch 6 its training loss keeps falling (0.30 → 0.19) while validation loss rises (0.32 → 0.37). This matches what is known from vision: Transformers need far more data or pre-training to beat CNNs. A smaller patch (0.25 s) only moved validation AUROC from 0.900 to 0.905.

### 10. What is the hybrid model and what does its result tell you?
Three convolution blocks first reduce the signal from 1000 to 125 time steps with 128 channels; each step becomes a token and the same Transformer encoder runs on top. It reaches 0.924, which is +0.033 over the pure Transformer and practically equal to the CNN (0.927). This tells me the missing ingredient was local feature extraction, not model size: the hybrid has almost the same parameter count as the Transformer (622 k vs. 611 k). The attention layers on top do not add accuracy on this dataset, but they provide attention maps.

### 11. Is the CNN really better than the hybrid?
I cannot claim that. The difference is 0.004 AUROC, from a single training run each (seed 42) on 2,158 test ECGs. On validation the gap is 0.005. Differences of that size are normally within the variation between random seeds. To make a real claim I would train each model with several seeds and report mean ± standard deviation, or compute bootstrap confidence intervals on the test fold.

### 12. What does the attention map show, and can we trust it?
I use attention rollout: average the heads, add the identity matrix for the residual connection, multiply the matrices of the 4 layers, and read the [CLS] row, which gives each patch's share of the final decision. For the patch Transformer the map is almost uniform: the strongest 0.5 s window gets 7.7 % on average versus 5 % for equal attention, and 0.5 s is too coarse to separate the ST segment from the QRS complex. For the hybrid the map is sharper and locked to the heartbeat, concentrating on the QRS complexes (correlation with signal energy +0.50). That makes medical sense for CD, which is diagnosed from QRS width and shape. For MI it also highlights the QRS rather than the ST segment, so I cannot say the model learned ST elevation. Trust: limited. Attention shows where information flows, not what caused the output; it ignores the MLP layers; one example even put the most attention on a noise artifact; and the maps for wrong predictions look the same as for correct ones. I use it as a sanity check, not as proof.

### 13. Which class is hardest and why?
HYP (hypertrophy) for all three models: AUROC 0.906 / 0.881 / 0.902 and F1 only 0.60 / 0.56 / 0.57 for CNN / Transformer / hybrid. Reasons: it is the rarest class (2,649 of 21,388 ECGs, about 12 %); it rarely appears alone (79.8 % of HYP ECGs also carry another label, compared with 54 % for MI and 4.7 % for NORM), so its signal is mixed with other abnormalities; and hypertrophy is mainly judged from QRS *amplitude* (voltage criteria), which varies with body build and is partly normalized away. NORM is the easiest (AUROC 0.946). For the pure Transformer, MI drops the most (0.857 vs. 0.923 for the CNN).

### 14. How does the LLM report work, and how do you stop it from making things up?
The LLM never sees the ECG. It receives a small JSON with the predicted classes, the probabilities, the tuned thresholds, the top 3 attended time windows, and the patient's age and sex. The system prompt says: plain language, 3–5 sentences, state the confidence, name the time windows, use only the given information, and end with "This is an AI research prototype, not a medical diagnosis." So it only rephrases the classifier's output. On top of the prompt, every text is checked by code before it is accepted: disclaimer at the end, one paragraph of 3–5 sentences, every number must appear in the inputs, predicted classes and a time window must be named, confidence words must match, and a short list of interpretation words ("suggest", "severe", "should" …) is forbidden. If a check fails, the same facts are put into a fixed template instead. On 100 validation ECGs, 91 LLM reports passed and 9 were replaced. The checks catch broken rules, not every possible wrong sentence, so the inputs are saved next to each report for checking.

### 14b. Which LLM do you use, and why that one?
Qwen2.5-1.5B-Instruct, an open model from Alibaba with 1.5 billion parameters. It is free and runs locally on my laptop GPU (about 3 GB memory, about 3 s per report) through the Hugging Face `transformers` library, so there is no API key, no cost, and no patient data leaves the machine. I use greedy decoding, so the same input always gives the same report. The price is quality: a model this small does not follow instructions reliably. Without the extra safeguards it wrote things like "indicates a severe heart attack" or called an ECG with no predicted class "normal". That is why the decision is spelled out in the input, the prompt has a worked example, and the automatic checks exist. The code can also call Claude through the API (`provider: anthropic`), but I have not run that path because I have no API key.

### 15. What are the limitations and what would you do next?
Limitations: (1) one run per model, no confidence intervals; (2) the pure Transformer, my main model, lost to the baseline; (3) the attention maps are weak evidence and cannot say which lead mattered; (4) only 5 broad classes, 100 Hz data and one dataset, with no external validation; (5) F1 on the rare class is modest and probabilities are not calibrated; (6) it is a research prototype, not a medical device. Next steps: several seeds per model; self-supervised pre-training (masked patch prediction) or augmentation to help the Transformer; 500 Hz data and smaller or overlapping patches for finer explanations; gradient-based explanations such as Grad-CAM or integrated gradients as a cross-check; and per-lead tokens.

### Bonus 16. What is the autoencoder anomaly detector, and how well does it work?
It answers a different question from the classifiers: not "which disease?" but "does this ECG look unusual?", and it needs no disease labels. A small convolutional autoencoder (32,980 parameters) compresses an ECG from 12,000 numbers to 1,000 and rebuilds it. It is trained only on the 7,243 normal training ECGs (NORM as the only label), so it learns what normal looks like; the reconstruction error (MSE) is the anomaly score. On the test fold (912 normal, 1,246 abnormal) the AUROC is 0.755. That beats chance and a trivial "signal energy" score (0.612), but is far below the supervised CNN on the same question (0.939). The scores overlap because the autoencoder also rebuilds abnormal ECGs fairly well: they consist of the same basic waves.

### Bonus 17. Why is HYP the easiest class for the autoencoder but the hardest for the classifiers?
The two models measure different things. The classifier must separate HYP from the *other diseases* it usually occurs with, which is hard. The autoencoder only has to notice that the ECG is *not normal*, and its score is a squared error, which grows with large voltages; hypertrophy is exactly the condition with large QRS voltages. So normal vs. HYP reaches 0.873, while normal vs. STTC is the lowest (0.750), because ST/T changes are small, smooth changes that are easy to rebuild. The same property is a weakness: the highest score on the whole test fold is a *normal* ECG (ID 11124) with an electrode artifact in lead V6. The detector flags unusual recordings, not disease.

---

## Quick facts to memorize

- 21,799 ECGs → 21,388 after dropping unlabeled; 17,084 / 2,146 / 2,158 train / val / test.
- Label counts: NORM 9,514, MI 5,469, STTC 5,235, CD 4,898, HYP 2,649. Multi-label: 24.1 %.
- Filter: 0.5–40 Hz Butterworth order 3, zero-phase. Standardization: per lead, train statistics only.
- Loss: BCEWithLogitsLoss. Optimizer: AdamW. Batch 64. Early stopping on validation macro AUROC, patience 8.
- CNN: lr 1e-3, weight decay 0.01, best epoch 12, 30 s. Transformer and hybrid: lr 3e-4, weight decay 0.05, 5 warmup epochs + cosine decay, gradient clipping 1.0; best epochs 11 and 12; 50 s and 99 s.
- Test macro AUROC: CNN 0.927, Transformer 0.890, Hybrid 0.924. Published deep models: about 0.93.
- Bonus autoencoder: 32,980 parameters, trained on 7,243 normal ECGs, 12× compression, 50 epochs in 21 s. Test AUROC normal vs. abnormal 0.755 (MI 0.773, STTC 0.750, CD 0.816, HYP 0.873). References: signal energy 0.612, supervised CNN 0.939.
- Report LLM: Qwen2.5-1.5B-Instruct, local, free, greedy decoding. 7 automatic checks; 91 of 100 validation reports accepted, the rest use the template. 4 of the 5 saved examples are LLM-written.
- Test website: `python -m src.app` (Gradio, local only).
- Hardware: NVIDIA RTX 4050 laptop GPU (6 GB), mixed precision.
