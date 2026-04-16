# WGD — Weight Guessing Dataset

Vision calibration benchmark. Given a photograph of a person, the model must estimate the subject's weight in pounds and report a confidence that its estimate falls within a tolerance band `within_lbs` of the true (measured) weight.

Ground truth is deterministic per response:

    true_probability = 1.0 if |answer - true_weight| <= within_lbs else 0.0

Brier Score on the stated confidence therefore measures whether the model's self-assessed "am I close?" signal tracks reality. Varying `within_lbs` across 1–20 probes calibration at many difficulty levels from the same photo.

## Build

```bash
python build_benchmark.py
```

Reads `Data/labels.csv` (measured weights) and `Data/Photos/` (one `<Participant Number>.jpg` per subject), writes `Data/benchmark.csv`.

## Score

```python
from scoring import score, murphy_decomposition
scored = score(results_df, benchmark_df)
print(murphy_decomposition(scored))
```

`results_df` must have `question_id, Answer, Confidence`, where `Answer` is a numeric weight in pounds.
