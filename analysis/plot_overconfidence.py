#!/usr/bin/env python3
"""RQ2: Mean overconfidence by task difficulty, baseline + SPD overlay."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.scoring import lifeeval_true_probability, medeval_true_probability  # noqa: E402

RESULTS = ROOT / "results"
FIG_DIR = ROOT / "analysis" / "figs" / "poster"

MODELS = [
    "anthropic_claude-haiku-4.5",
    "google_gemini-2.5-flash",
    "meta-llama_llama-4-maverick",
    "openai_gpt-5.4-mini",
]
MODEL_LABELS = ["Claude Haiku 4.5", "Gemini 2.5 Flash", "Llama 4 Maverick", "GPT-5.4 Mini"]
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MODEL_COLORS = dict(zip(MODELS, CB_PALETTE))


def dedup_columns(df):
    renames = {}
    drops = []
    for col in df.columns:
        if col.endswith("_x"):
            base = col[:-2]
            renames[col] = base
            if f"{base}_y" in df.columns:
                drops.append(f"{base}_y")
    df = df.rename(columns=renames)
    return df.drop(columns=drops, errors="ignore")


def load_domain(domain):
    frames = []
    for model in MODELS:
        df = dedup_columns(pd.read_csv(RESULTS / domain / f"{model}.csv"))
        df["model"] = model
        df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def score_wgd(df):
    answer = pd.to_numeric(df["Answer"], errors="coerce")
    df["correct"] = ((answer - df["true_weight"].astype(float)).abs() <= df["within_lbs"].astype(float)).astype(float)
    df.loc[answer.isna(), "correct"] = np.nan
    df["overconfidence"] = df["Confidence"] - df["correct"]
    return df


def score_lifeeval(df):
    answer = pd.to_numeric(df["Answer"], errors="coerce")
    valid = answer.notna() & df["sex"].notna()
    df["true_probability"] = np.nan
    df.loc[valid, "true_probability"] = [
        lifeeval_true_probability(a, age, sex, r)
        for a, age, sex, r in zip(answer[valid], df.loc[valid, "min_age"],
                                   df.loc[valid, "sex"], df.loc[valid, "radius"])
    ]
    df["overconfidence"] = df["Confidence"] - df["true_probability"]
    return df


def score_medeval(df):
    valid = df["Answer"].notna() & df["differential_json"].notna()
    df["true_probability"] = np.nan
    df.loc[valid, "true_probability"] = [
        medeval_true_probability(a, d)
        for a, d in zip(df.loc[valid, "Answer"], df.loc[valid, "differential_json"])
    ]
    df["overconfidence"] = df["Confidence"] - df["true_probability"]
    return df


def main():
    print("Loading & scoring SPD...")
    wgd_spd = score_wgd(load_domain("WGD_SPD"))
    le_spd = score_lifeeval(load_domain("LifeEval_SPD"))
    me_spd = score_medeval(load_domain("MedEval_SPD"))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "axes.linewidth": 0.6,
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

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4), sharey=True, layout="constrained")

    configs = [
        (axes[0], "WGD", wgd_spd, "within_lbs", "Radius (lbs)", range(1, 21, 2)),
        (axes[1], "LifeEval", le_spd, "radius", "Radius (years)", range(1, 21, 2)),
        (axes[2], "MedEval", me_spd, "removal_pct", "% Candidates Removed", [0, 10, 25, 50]),
    ]

    for ax, title, spd_df, diff_col, xlabel, xticks in configs:
        for model, label in zip(MODELS, MODEL_LABELS):
            color = MODEL_COLORS[model]
            sub_spd = spd_df[spd_df["model"] == model]
            grouped_spd = sub_spd.groupby(diff_col)["overconfidence"].mean()
            ax.plot(grouped_spd.index, grouped_spd.values, marker="o", markersize=3,
                    color=color, linewidth=1.2, label=label)

        ax.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xticks(xticks)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean Overconfidence")

    axes[0].legend(loc="upper right")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "overconfidence_by_difficulty.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
