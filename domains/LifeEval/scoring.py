"""
Brier scoring for LifeEval.

Given a results CSV with columns (question_id, Answer, Confidence) and the
benchmark CSV (with min_age, sex, radius), compute:
    true_probability = P(death in [Answer-radius, Answer+radius] | survived to min_age)
    brier            = (Confidence - true_probability)^2

True probability is computed via the Gompertz conditional survival CDF,
fit to the 2022 US Period Life Table (ages 5-94, separate male/female).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GompertzParams:
    """Gompertz hazard parameters: h(x) = b * exp(c * x)"""
    b: float
    c: float


def fit_gompertz_to_life_table(life_table_path: str, sex: str = "male") -> GompertzParams:
    """Fit Gompertz hazard via MLE on ages 5-94 of the period life table."""
    df = pd.read_csv(life_table_path)
    col_prefix = "MALE" if sex.lower() == "male" else "FEMALE"
    q_col = f"Death probability ({col_prefix})"

    ages = df["Age"].values.astype(float)
    qx = df[q_col].values.astype(float)

    mask = (ages >= 5) & (ages <= 94)
    fit_ages = ages[mask]
    fit_qx = qx[mask]

    def neg_log_likelihood(params):
        log_b, c = params
        b = np.exp(log_b)
        if c <= 0:
            return 1e12
        exponent = -(b / c) * (np.exp(c * (fit_ages + 1)) - np.exp(c * fit_ages))
        predicted_qx = 1.0 - np.exp(exponent)
        predicted_qx = np.clip(predicted_qx, 1e-15, 1 - 1e-15)
        ll = fit_qx * np.log(predicted_qx) + (1 - fit_qx) * np.log(1 - predicted_qx)
        return -np.sum(ll)

    result = minimize(
        neg_log_likelihood,
        x0=[np.log(1e-5), 0.085],
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12},
    )
    return GompertzParams(b=np.exp(result.x[0]), c=result.x[1])


def _conditional_survival(x: float, a: float, params: GompertzParams) -> float:
    """S(x | X >= a) = exp(-(b/c)(exp(cx) - exp(ca)))"""
    b, c = params.b, params.c
    return float(np.exp(-(b / c) * (np.exp(c * x) - np.exp(c * a))))


def _window_probability(y: float, a: float, r: float, params: GompertzParams) -> float:
    """P(death in [y-r, y+r] | survived to a), closed-form via conditional survival CDF."""
    lo = max(y - r, a)
    hi = y + r
    if lo >= hi:
        return 0.0
    return _conditional_survival(lo, a, params) - _conditional_survival(hi, a, params)


_LIFE_TABLE = Path(__file__).parent / "Data" / "PeriodLifeTable_2022_RawData.csv"
_PARAMS: dict[str, GompertzParams] | None = None


def _get_params() -> dict[str, GompertzParams]:
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = {
            "male": fit_gompertz_to_life_table(str(_LIFE_TABLE), "male"),
            "female": fit_gompertz_to_life_table(str(_LIFE_TABLE), "female"),
        }
    return _PARAMS


def true_probability(answer: float, min_age: float, sex: str, radius: float) -> float:
    params = _get_params()[sex.lower()]
    return _window_probability(answer, min_age, radius, params)


def score(results: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    df = results.merge(
        benchmark[["question_id", "min_age", "sex", "radius"]],
        on="question_id",
        how="left",
    )
    answer = pd.to_numeric(df["Answer"], errors="coerce")
    df["true_probability"] = [
        true_probability(a, age, sex, r)
        for a, age, sex, r in zip(answer, df["min_age"], df["sex"], df["radius"])
    ]
    conf = pd.to_numeric(df["Confidence"], errors="coerce")
    df["brier"] = (conf - df["true_probability"]) ** 2
    return df


def murphy_decomposition(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """BS = Reliability - Resolution + Uncertainty."""
    conf = pd.to_numeric(df["Confidence"], errors="coerce").to_numpy()
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
