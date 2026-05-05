#!/usr/bin/env python3
"""Separate Generic vs SPD illustration plots for a LifeEval question."""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.scoring import _get_gompertz_params, GompertzParams  # noqa: E402

GENERIC_BENCHMARK = ROOT / "domains" / "LifeEval" / "Data" / "benchmark.csv"
GENERIC_RESULTS = ROOT / "results" / "LifeEval"
SPD_RESULTS = ROOT / "results" / "LifeEval_SPD"
FIG_DIR = ROOT / "analysis" / "figs" / "poster"

VALID_BIN_WIDTHS = {2, 10, 20, 40}
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def conditional_pdf(x: np.ndarray, a: float, params: GompertzParams) -> np.ndarray:
    b, c = params.b, params.c
    hazard = b * np.exp(c * x)
    log_survival = -(b / c) * (np.exp(c * x) - np.exp(c * a))
    return hazard * np.exp(log_survival)


def parse_all_bins(raw) -> list[tuple[float, float, float]]:
    if not raw or (isinstance(raw, float) and np.isnan(raw)):
        return []
    match = _JSON_ARRAY_RE.search(str(raw))
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    bins = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        try:
            lo, hi, conf = float(item["min"]), float(item["max"]), float(item["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if hi > lo:
            bins.append((lo, hi, conf))
    return sorted(bins, key=lambda b: b[0])


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.color": "#333333",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.grid": False,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def main():
    parser = argparse.ArgumentParser(description="Generic vs SPD illustration plots for LifeEval")
    parser.add_argument("--question-id", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bin-width", type=int, choices=[2, 10, 20, 40], default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    bench = pd.read_csv(GENERIC_BENCHMARK)
    row = bench[bench["question_id"] == args.question_id]
    if row.empty:
        sys.exit(f"Error: question_id {args.question_id} not found in benchmark")
    row = row.iloc[0]
    min_age, sex, radius = int(row["min_age"]), row["sex"], int(row["radius"])

    bin_width = args.bin_width
    if bin_width is None:
        candidate = 2 * radius
        if candidate in VALID_BIN_WIDTHS:
            bin_width = candidate
        else:
            sys.exit(
                f"Error: radius={radius} has no natural SPD match "
                f"(2*radius={candidate} not in {sorted(VALID_BIN_WIDTHS)}). "
                f"Specify --bin-width explicitly."
            )

    gen_path = GENERIC_RESULTS / f"{args.model}.csv"
    if not gen_path.exists():
        sys.exit(f"Error: {gen_path} not found")
    gen_df = pd.read_csv(gen_path)
    gen_row = gen_df[gen_df["question_id"] == args.question_id]
    if gen_row.empty:
        sys.exit(f"Error: question_id {args.question_id} not in {gen_path.name}")
    gen_row = gen_row.iloc[0]
    gen_answer = float(gen_row["Answer"])

    spd_qid = f"le_spd_{min_age}_{sex}_{bin_width}"
    spd_path = SPD_RESULTS / f"{args.model}.csv"
    if not spd_path.exists():
        sys.exit(f"Error: {spd_path} not found")
    spd_df = pd.read_csv(spd_path)
    spd_row = spd_df[spd_df["question_id"] == spd_qid]
    if spd_row.empty:
        sys.exit(f"Error: SPD question {spd_qid} not in {spd_path.name}")
    spd_row = spd_row.iloc[0]
    bins = parse_all_bins(spd_row["raw"])
    if not bins:
        sys.exit(f"Error: no valid bins parsed for {spd_qid}")

    params = _get_gompertz_params()[sex]
    x = np.linspace(max(min_age, 1), 110, 2000)
    pdf = conditional_pdf(x, float(min_age), params)
    modal_idx = max(range(len(bins)), key=lambda i: bins[i][2])

    # Shared axis limits
    all_edges = [b[0] for b in bins] + [b[1] for b in bins] + [gen_answer - radius, gen_answer + radius]
    xlim = (min(min(all_edges), x[0]) - 2, max(max(all_edges), x[-1]) + 2)
    ylim = (0, max(pdf) * 1.1)

    setup_style()
    stem = args.output or "generic_vs_spd"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Generic plot ---
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.plot(x, pdf, color="#333333", linewidth=1.5, zorder=3)
    ax.fill_between(x, pdf, alpha=0.06, color="#333333", zorder=1)

    win_lo = max(gen_answer - radius, min_age)
    win_hi = gen_answer + radius
    mask = (x >= win_lo) & (x <= win_hi)
    ax.fill_between(x[mask], pdf[mask], alpha=0.45, color="#FFD700", zorder=2)
    ax.axvline(gen_answer, color="#D32F2F", linewidth=2, zorder=4)

    ax.set_xlabel("Age at death (years)")
    ax.set_ylabel("Probability density")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.savefig(FIG_DIR / f"{stem}_generic.png", dpi=300)
    print(f"Saved: {FIG_DIR / f'{stem}_generic.png'}")
    plt.close(fig)

    # --- SPD plot ---
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.plot(x, pdf, color="#333333", linewidth=1.5, zorder=3)
    ax.fill_between(x, pdf, alpha=0.06, color="#333333", zorder=1)

    for i, (lo, hi, conf) in enumerate(bins):
        density = conf / (hi - lo)
        is_modal = i == modal_idx
        color = "#FFD700" if is_modal else "#89CFF0"
        alpha = 0.7 if is_modal else 0.5
        ax.bar((lo + hi) / 2, density, width=(hi - lo), alpha=alpha,
               color=color, edgecolor="white", linewidth=0.5, zorder=2)

    ax.set_xlabel("Age at death (years)")
    ax.set_ylabel("Probability density")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.savefig(FIG_DIR / f"{stem}_spd.png", dpi=300)
    print(f"Saved: {FIG_DIR / f'{stem}_spd.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
