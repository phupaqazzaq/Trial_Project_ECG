# Final results on the PTB-XL test fold (fold 10)

| Model | Parameters | Train time | Macro AUROC | AUROC NORM | AUROC MI | AUROC STTC | AUROC CD | AUROC HYP | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| 1D-CNN (baseline) | 190,757 | 30 s | **0.9272** | 0.946 | 0.923 | 0.939 | 0.922 | 0.906 | 0.748 |
| Patch Transformer | 610,565 | 50 s | **0.8901** | 0.918 | 0.857 | 0.915 | 0.880 | 0.881 | 0.687 |
| Hybrid CNN + Transformer | 622,117 | 99 s | **0.9235** | 0.945 | 0.919 | 0.931 | 0.920 | 0.902 | 0.735 |

- Test fold = official PTB-XL fold 10 (2,158 ECGs), used once per model.
- Thresholds for F1 were tuned per class on the validation fold (fold 9).
- Train time: one run on an NVIDIA RTX 4050 laptop GPU, seed 42, with early stopping.
- Published deep models reach about 0.93 macro AUROC on this task.
