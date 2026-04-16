"""
Brier scoring for WGD.

Ground truth is deterministic per (photo, within_lbs, model_answer):
    true_probability = 1.0 if |answer - true_weight| <= within_lbs else 0.0
    brier            = (Confidence - true_probability)^2
"""

import numpy as np
import pandas as pd


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def score(results: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    df = results.merge(
        benchmark[["question_id", "within_lbs", "true_weight"]],
        on="question_id",
        how="left",
    )
    answer = df["Answer"].map(_to_float)
    conf = df["Confidence"].map(_to_float)
    within = df["within_lbs"].astype(float)
    true_w = df["true_weight"].astype(float)

    df["true_probability"] = ((answer - true_w).abs() <= within).astype(float)
    df.loc[answer.isna(), "true_probability"] = np.nan
    df["brier"] = (conf - df["true_probability"]) ** 2
    return df


def murphy_decomposition(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """BS = Reliability - Resolution + Uncertainty."""
    conf = df["Confidence"].map(_to_float).to_numpy()
    p = df["true_probability"].to_numpy(dtype=float)
    mask = ~(np.isnan(conf) | np.isnan(p))
    conf, p = conf[mask], p[mask]
    if conf.size == 0:
        return {"reliability": np.nan, "resolution": np.nan,
                "uncertainty": np.nan, "brier": np.nan}
    p_bar = p.mean()
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    reliability = resolution = 0.0
    for b in range(n_bins):
        idx = bins == b
        if not idx.any():
            continue
        w = idx.sum() / conf.size
        reliability += w * (conf[idx].mean() - p[idx].mean()) ** 2
        resolution += w * (p[idx].mean() - p_bar) ** 2
    uncertainty = float((p * (1 - p)).mean())
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "brier": float(((conf - p) ** 2).mean()),
    }
