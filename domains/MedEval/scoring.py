"""
Brier scoring for MedEval.

Given a results CSV with columns (question_id, Answer, Confidence) and the
benchmark CSV (with differential_json), compute:
    true_probability = differential[Answer]   (0.0 if Answer not in differential)
    brier            = (Confidence - true_probability)^2

Match is case-insensitive and whitespace-insensitive on pathology name.
"""

import json

import numpy as np
import pandas as pd


def _normalize(s: str) -> str:
    return "".join(c.lower() for c in str(s) if not c.isspace())


def true_probability(answer: str, differential_json: str) -> float:
    try:
        differential = json.loads(differential_json)
    except (TypeError, json.JSONDecodeError):
        return 0.0
    norm = _normalize(answer)
    for pathology, prob in differential:
        if _normalize(pathology) == norm:
            return float(prob)
    return 0.0


def score(results: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    df = results.merge(
        benchmark[["question_id", "true_pathology", "differential_json"]],
        on="question_id",
        how="left",
    )
    df["true_probability"] = [
        true_probability(a, d)
        for a, d in zip(df["Answer"], df["differential_json"])
    ]
    conf = pd.to_numeric(df["Confidence"], errors="coerce")
    df["brier"] = (conf - df["true_probability"]) ** 2
    return df


def murphy_decomposition(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """BS = Reliability - Resolution + Uncertainty."""
    conf = pd.to_numeric(df["Confidence"], errors="coerce").to_numpy()
    p = df["true_probability"].to_numpy(dtype=float)
    mask = ~np.isnan(conf)
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
        c_b = conf[idx].mean()
        p_b = p[idx].mean()
        reliability += w * (c_b - p_b) ** 2
        resolution += w * (p_b - p_bar) ** 2
    uncertainty = float((p * (1 - p)).mean())
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "brier": float(((conf - p) ** 2).mean()),
    }
