# LifeEval

Actuarial mortality calibration benchmark using the 2022 US Period Life Table and the Gompertz mortality model.

Each question presents a conditional survival age and sex, asking the model to estimate age-at-death plus a confidence that the estimate is within a given radius. Ground truth is the Gompertz conditional survival CDF: `true_probability = P(death in [Answer-r, Answer+r] | survived to age a)`, computed in closed form.

Scoring uses the Brier Score: `(confidence - true_probability)^2`.

4040 questions: 101 ages (0-100) x 2 sexes x 20 radii (1-20 years).

## Build

```bash
python build_benchmark.py
```

Produces `Data/benchmark.csv` with columns:
`question_prompt, confidence_prompt, true_lifespan, question_id, min_age, sex, radius, best_answer, MAS, gold_response`.

## Score

Given a results CSV with `question_id, Answer, Confidence`:

```python
from scoring import score, murphy_decomposition
scored = score(results_df, benchmark_df)
print(murphy_decomposition(scored))
```
