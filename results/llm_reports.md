# Example ECG reports

Model: `hybrid`. Each report is generated only from the inputs listed above it.

## ECG 9 (true labels: NORM)

**Inputs given to the report generator**

- Patient: 55, male
- Predicted classes: NORM
- Probabilities: NORM 0.96, MI 0.01, STTC 0.00, CD 0.04, HYP 0.00
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 2.5-3.0 s, 3.5-4.0 s, 9.5-10.0 s

**Report** (source: template (no API key set))

> For this ECG (male patient, age 55), the model predicts: normal ECG (probability 0.96, threshold 0.39, high confidence). The other classes stayed below their thresholds (MI 0.01, STTC 0.00, CD 0.04, HYP 0.00). The time segments that influenced the result most were 2.5-3.0 s, 3.5-4.0 s, 9.5-10.0 s of the 10-second recording. This is an AI research prototype, not a medical diagnosis.

## ECG 162 (true labels: MI, STTC, CD, HYP)

**Inputs given to the report generator**

- Patient: 85, female
- Predicted classes: MI, STTC, CD, HYP
- Probabilities: NORM 0.00, MI 0.65, STTC 0.51, CD 0.99, HYP 0.65
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 9.5-10.0 s, 8.0-8.5 s, 3.0-3.5 s

**Report** (source: template (no API key set))

> For this ECG (female patient, age 85), the model predicts: myocardial infarction (probability 0.65, threshold 0.38, moderate confidence); ST/T change (probability 0.51, threshold 0.36, moderate confidence); conduction disturbance (probability 0.99, threshold 0.59, high confidence); hypertrophy (probability 0.65, threshold 0.50, moderate confidence). The other classes stayed below their thresholds (NORM 0.00). The time segments that influenced the result most were 9.5-10.0 s, 8.0-8.5 s, 3.0-3.5 s of the 10-second recording. This is an AI research prototype, not a medical diagnosis.

## ECG 128 (true labels: STTC)

**Inputs given to the report generator**

- Patient: 33, male
- Predicted classes: STTC
- Probabilities: NORM 0.14, MI 0.12, STTC 0.56, CD 0.00, HYP 0.00
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 0.0-0.5 s, 3.5-4.0 s, 5.5-6.0 s

**Report** (source: template (no API key set))

> For this ECG (male patient, age 33), the model predicts: ST/T change (probability 0.56, threshold 0.36, moderate confidence). The other classes stayed below their thresholds (NORM 0.14, MI 0.12, CD 0.00, HYP 0.00). The time segments that influenced the result most were 0.0-0.5 s, 3.5-4.0 s, 5.5-6.0 s of the 10-second recording. This is an AI research prototype, not a medical diagnosis.

## ECG 65 (true labels: CD)

**Inputs given to the report generator**

- Patient: 62, male
- Predicted classes: NORM, CD
- Probabilities: NORM 0.73, MI 0.00, STTC 0.00, CD 0.85, HYP 0.02
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 0.0-0.5 s, 1.0-1.5 s, 6.0-6.5 s

**Report** (source: template (no API key set))

> For this ECG (male patient, age 62), the model predicts: normal ECG (probability 0.73, threshold 0.39, moderate confidence); conduction disturbance (probability 0.85, threshold 0.59, high confidence). The other classes stayed below their thresholds (MI 0.00, STTC 0.00, HYP 0.02). The time segments that influenced the result most were 0.0-0.5 s, 1.0-1.5 s, 6.0-6.5 s of the 10-second recording. This is an AI research prototype, not a medical diagnosis.

## ECG 271 (true labels: MI, STTC, HYP)

**Inputs given to the report generator**

- Patient: 75, male
- Predicted classes: MI, STTC, HYP
- Probabilities: NORM 0.00, MI 0.97, STTC 0.94, CD 0.14, HYP 0.61
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 6.0-6.5 s, 4.0-4.5 s, 2.0-2.5 s

**Report** (source: template (no API key set))

> For this ECG (male patient, age 75), the model predicts: myocardial infarction (probability 0.97, threshold 0.38, high confidence); ST/T change (probability 0.94, threshold 0.36, high confidence); hypertrophy (probability 0.61, threshold 0.50, moderate confidence). The other classes stayed below their thresholds (NORM 0.00, CD 0.14). The time segments that influenced the result most were 6.0-6.5 s, 4.0-4.5 s, 2.0-2.5 s of the 10-second recording. This is an AI research prototype, not a medical diagnosis.
