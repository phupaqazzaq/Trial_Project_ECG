# Example ECG reports

Model: `hybrid`. Each report is generated only from the inputs listed above it.

## ECG 9 (true labels: NORM)

**Inputs given to the report generator**

- Patient: 55, male
- Predicted classes: NORM
- Probabilities: NORM 0.96, MI 0.01, STTC 0.00, CD 0.04, HYP 0.00
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 2.5-3.0 s, 3.5-4.0 s, 9.5-10.0 s

**Report** (source: LLM (Qwen/Qwen2.5-1.5B-Instruct), passed the automatic checks)

> For this 55-year-old male patient, the model has classified the ECG as normal (NORM) with high confidence. The probabilities show that the NORM class reaches its decision threshold at 0.96, while all other classes have lower probabilities and did not reach their respective thresholds. The top windows that influenced the result are 2.5-3.0 s, 3.5-4.0 s, and 9.5-10.0 s. This is an AI research prototype, not a medical diagnosis.

## ECG 162 (true labels: MI, STTC, CD, HYP)

**Inputs given to the report generator**

- Patient: 85, female
- Predicted classes: MI, STTC, CD, HYP
- Probabilities: NORM 0.00, MI 0.65, STTC 0.51, CD 0.99, HYP 0.65
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 9.5-10.0 s, 8.0-8.5 s, 3.0-3.5 s

**Report** (source: template (LLM text rejected: more than one paragraph (line breaks or a list)))

> For this ECG (female patient, age 85), the model predicts: myocardial infarction (probability 0.65, threshold 0.38, moderate confidence); ST/T change (probability 0.51, threshold 0.36, moderate confidence); conduction disturbance (probability 0.99, threshold 0.59, high confidence); hypertrophy (probability 0.65, threshold 0.50, moderate confidence). The other classes stayed below their thresholds (NORM 0.00). The time segments that influenced the result most were 9.5-10.0 s, 8.0-8.5 s, 3.0-3.5 s of the 10-second recording. This is an AI research prototype, not a medical diagnosis.

## ECG 128 (true labels: STTC)

**Inputs given to the report generator**

- Patient: 33, male
- Predicted classes: STTC
- Probabilities: NORM 0.14, MI 0.12, STTC 0.56, CD 0.00, HYP 0.00
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 0.0-0.5 s, 3.5-4.0 s, 5.5-6.0 s

**Report** (source: LLM (Qwen/Qwen2.5-1.5B-Instruct), passed the automatic checks)

> For this 33-year-old male patient, the model predicts ST/T change (STTC) with moderate confidence. The prediction was made based on probabilities where STTC has a moderate confidence level of 0.56, which exceeds the decision threshold of 0.36. There are no other classes reaching their respective thresholds. The top influencing windows according to the model were 0.0-0.5 s, 3.5-4.0 s, and 5.5-6.0 s. This is an AI research prototype, not a medical diagnosis.

## ECG 65 (true labels: CD)

**Inputs given to the report generator**

- Patient: 62, male
- Predicted classes: NORM, CD
- Probabilities: NORM 0.73, MI 0.00, STTC 0.00, CD 0.85, HYP 0.02
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 0.0-0.5 s, 1.0-1.5 s, 6.0-6.5 s

**Report** (source: LLM (Qwen/Qwen2.5-1.5B-Instruct), passed the automatic checks)

> For this 62-year-old male patient, the model predicts a normal ECG (NORM) with moderate confidence and a conduction disturbance (CD) with high confidence. The probabilities are as follows: NORM has a moderate confidence of 0.73, CD has a high confidence of 0.85, while STTC and HYP have very low probabilities of 0.0 and 0.02 respectively. Both NORM and CD surpassed their respective decision thresholds. The top windows influencing the prediction were 0.0-0.5 s, 1.0-1.5 s, and 6.0-6.5 s. This is an AI research prototype, not a medical diagnosis.

## ECG 271 (true labels: MI, STTC, HYP)

**Inputs given to the report generator**

- Patient: 75, male
- Predicted classes: MI, STTC, HYP
- Probabilities: NORM 0.00, MI 0.97, STTC 0.94, CD 0.14, HYP 0.61
- Thresholds: NORM 0.39, MI 0.38, STTC 0.36, CD 0.59, HYP 0.50
- Top attended windows: 6.0-6.5 s, 4.0-4.5 s, 2.0-2.5 s

**Report** (source: LLM (Qwen/Qwen2.5-1.5B-Instruct), passed the automatic checks)

> For this 75-year-old male patient, the model predicts three conditions: myocardial infarction (MI), ST/T change (STTC), and hypertrophy (HYP). The MI has high confidence due to probabilities above the decision threshold, while STTC also has high confidence but less than MI. Hypertrophy has moderate confidence. The top influencing windows are 6.0-6.5 s, 4.0-4.5 s, and 2.0-2.5 s. This is an AI research prototype, not a medical diagnosis.
