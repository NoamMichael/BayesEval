# MedEval

Differential-diagnosis calibration benchmark built on [DDXPlus](https://arxiv.org/abs/2205.09148).

Each question presents a synthetic patient (age, sex, translated symptom list in JSON) alongside a shuffled list of candidate pathologies drawn from the DDXPlus differential. The model picks a single diagnosis and reports a confidence in [0, 1]. Ground truth is the DDXPlus differential diagnosis — a per-patient probability distribution over pathologies — so `true_probability = differential[model_answer]`.

Scoring uses the Brier Score: `(confidence - true_probability)^2`.

## Build

```bash
python build_benchmark.py --ddxplus ../../../ddxplus/22687585 --split test --n 500
```

Produces `Data/benchmark.csv` with columns:
`question_id, question_prompt, confidence_prompt, age, sex, true_pathology, differential_json`.

## Context-Removal Variants

To study calibration under reduced symptom context, the benchmark supports generating
variants with a fraction of findings randomly removed per question.

Questions with fewer than 10 symptoms are excluded. For each remaining question,
`ceil(n_findings * pct / 100)` findings are removed (rounding up so the removal
is at least as aggressive as requested — e.g. 11 findings at 50% removes 6).

```bash
python build_benchmark.py --ddxplus ../../../ddxplus/22687585 --split test --n 500 \
    --min-findings 10 --removal-pcts 0,10,25,50
```

Produces per-variant files (`benchmark_remove{pct}.csv`) and a combined
`benchmark_combined.csv` with an extra `removal_pct` column for evaluation.

## Score

Given a results CSV with `question_id, Answer, Confidence`:

```python
from scoring import score, murphy_decomposition
scored = score(results_df, benchmark_df)
print(murphy_decomposition(scored))
```
