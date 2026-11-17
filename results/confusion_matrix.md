# Confusion matrices on the PTB-XL test fold (fold 10)

2,158 test ECGs. Each class uses the threshold tuned on the validation fold.
TP = has it, predicted yes. FP = does not have it, predicted yes (false alarm).
FN = has it, predicted no (missed). TN = does not have it, predicted no.

## 1D-CNN (baseline)

| Class | Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| NORM | 0.43 | 898 | 220 | 65 | 975 | 0.803 | 0.933 | 0.863 |
| MI | 0.35 | 428 | 164 | 122 | 1,444 | 0.723 | 0.778 | 0.750 |
| STTC | 0.25 | 435 | 174 | 86 | 1,463 | 0.714 | 0.835 | 0.770 |
| CD | 0.56 | 379 | 125 | 117 | 1,537 | 0.752 | 0.764 | 0.758 |
| HYP | 0.17 | 158 | 105 | 104 | 1,791 | 0.601 | 0.603 | 0.602 |

Macro F1 0.749. All 5 answers right for 1,282 of 2,158 ECGs (59.4 %).

## Patch Transformer

| Class | Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| NORM | 0.32 | 885 | 275 | 78 | 920 | 0.763 | 0.919 | 0.834 |
| MI | 0.29 | 383 | 243 | 167 | 1,365 | 0.612 | 0.696 | 0.651 |
| STTC | 0.29 | 401 | 189 | 120 | 1,448 | 0.680 | 0.770 | 0.722 |
| CD | 0.53 | 293 | 88 | 203 | 1,574 | 0.769 | 0.591 | 0.668 |
| HYP | 0.22 | 157 | 142 | 105 | 1,754 | 0.525 | 0.599 | 0.560 |

Macro F1 0.687. All 5 answers right for 1,140 of 2,158 ECGs (52.8 %).

## Hybrid CNN + Transformer

| Class | Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| NORM | 0.39 | 897 | 222 | 66 | 973 | 0.802 | 0.931 | 0.862 |
| MI | 0.38 | 390 | 136 | 160 | 1,472 | 0.741 | 0.709 | 0.725 |
| STTC | 0.36 | 432 | 175 | 89 | 1,462 | 0.712 | 0.829 | 0.766 |
| CD | 0.59 | 345 | 72 | 151 | 1,590 | 0.827 | 0.696 | 0.756 |
| HYP | 0.50 | 130 | 65 | 132 | 1,831 | 0.667 | 0.496 | 0.569 |

Macro F1 0.736. All 5 answers right for 1,340 of 2,158 ECGs (62.1 %).
