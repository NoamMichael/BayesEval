# MedEval

Differential-diagnosis calibration benchmark built on [DDXPlus](https://arxiv.org/abs/2205.09148).

Each question presents a synthetic patient (age, sex, translated symptom list) and asks the model for its single most likely pathology plus a confidence in [0, 1]. Ground truth is the DDXPlus differential diagnosis — a per-patient probability distribution over pathologies — so `true_probability = differential[model_answer]`, mirroring LifeEval's answer-dependent ground truth.

Scoring uses the Brier Score: `(confidence - true_probability)^2`.

## Build

```bash
python build_benchmark.py --ddxplus ../../../ddxplus/22687585 --split test --n 500
```

Produces `Data/benchmark.csv` with columns:
`question_id, question_prompt, confidence_prompt, age, sex, true_pathology, differential_json`.

## Score

Given a results CSV with `question_id, Answer, Confidence`:

```python
from scoring import score, murphy_decomposition
scored = score(results_df, benchmark_df)
print(murphy_decomposition(scored))
```
