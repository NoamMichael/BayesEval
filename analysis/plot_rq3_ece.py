#!/usr/bin/env python3
"""RQ3 grouped bar chart: Baseline vs SPD ECE with bootstrap significance."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.scoring import (  # noqa: E402
    lifeeval_true_probability,
    medeval_true_probability,
)

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

BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
N_BOOT = 2000
RNG = np.random.default_rng(42)


def compute_ece(conf, outcome):
    bins = pd.cut(conf, bins=BIN_EDGES, right=False, include_lowest=True)
    grouped = pd.DataFrame({"conf": conf.values, "outcome": outcome.values, "bin": bins.values}).groupby("bin", observed=False)
    n = grouped.size()
    total = n.sum()
    if total == 0:
        return np.nan
    return (n / total * (grouped["outcome"].mean() - grouped["conf"].mean()).abs()).sum()


def bootstrap_ece(conf, outcome, n_boot=N_BOOT):
    n = len(conf)
    eces = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        eces[i] = compute_ece(conf.iloc[idx].reset_index(drop=True),
                              outcome.iloc[idx].reset_index(drop=True))
    return eces


def dedup_columns(df):
    """Fix double-merged columns: prefer _x suffix, drop _y."""
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


def load_domain(domain, spd_domain):
    frames_base, frames_spd = [], []
    for model in MODELS:
        b = dedup_columns(pd.read_csv(RESULTS / domain / f"{model}.csv"))
        b["model"] = model
        frames_base.append(b)
        s = dedup_columns(pd.read_csv(RESULTS / spd_domain / f"{model}.csv"))
        s["model"] = model
        frames_spd.append(s)
    return pd.concat(frames_base, ignore_index=True), pd.concat(frames_spd, ignore_index=True)


def add_outcomes_wgd(df):
    answer = pd.to_numeric(df["Answer"], errors="coerce")
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    df["outcome"] = ((answer - df["true_weight"].astype(float)).abs() <= df["within_lbs"].astype(float)).astype(float)
    df.loc[answer.isna(), "outcome"] = np.nan
    return df


def add_outcomes_lifeeval(df):
    answer = pd.to_numeric(df["Answer"], errors="coerce")
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    valid = answer.notna() & df["sex"].notna()
    df["outcome"] = np.nan
    df.loc[valid, "outcome"] = [
        lifeeval_true_probability(a, age, sex, r)
        for a, age, sex, r in zip(answer[valid], df.loc[valid, "min_age"],
                                   df.loc[valid, "sex"], df.loc[valid, "radius"])
    ]
    return df


def add_outcomes_medeval(df):
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    valid = df["Answer"].notna() & df["differential_json"].notna()
    df["outcome"] = np.nan
    df.loc[valid, "outcome"] = [
        medeval_true_probability(a, d)
        for a, d in zip(df.loc[valid, "Answer"], df.loc[valid, "differential_json"])
    ]
    return df


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def two_sided_p(boot_delta):
    p_lower = (boot_delta <= 0).mean()
    return 2 * min(p_lower, 1 - p_lower)


def main():
    # Load and score all data
    wgd_base, wgd_spd = load_domain("WGD", "WGD_SPD")
    le_base, le_spd = load_domain("LifeEval", "LifeEval_SPD")
    me_base, me_spd = load_domain("MedEval", "MedEval_SPD")

    print("Scoring WGD...")
    wgd_base, wgd_spd = add_outcomes_wgd(wgd_base), add_outcomes_wgd(wgd_spd)
    print("Scoring LifeEval...")
    le_base, le_spd = add_outcomes_lifeeval(le_base), add_outcomes_lifeeval(le_spd)
    print("Scoring MedEval...")
    me_base, me_spd = add_outcomes_medeval(me_base), add_outcomes_medeval(me_spd)

    domains = [
        ("WGD", wgd_base, wgd_spd),
        ("LifeEval", le_base, le_spd),
        ("MedEval", me_base, me_spd),
    ]

    # Compute ECEs and bootstrap
    results = {}
    for domain_name, base_df, spd_df in domains:
        for model in MODELS:
            sb = base_df[base_df["model"] == model].dropna(subset=["Confidence", "outcome"])
            ss = spd_df[spd_df["model"] == model].dropna(subset=["Confidence", "outcome"])

            ece_base = compute_ece(sb["Confidence"], sb["outcome"])
            ece_spd = compute_ece(ss["Confidence"], ss["outcome"])

            print(f"  Bootstrapping {domain_name} / {model}...")
            boot_base = bootstrap_ece(sb["Confidence"], sb["outcome"])
            boot_spd = bootstrap_ece(ss["Confidence"], ss["outcome"])
            boot_delta = boot_base - boot_spd

            p_value = two_sided_p(boot_delta)

            results[(domain_name, model)] = {
                "ece_base": ece_base,
                "ece_spd": ece_spd,
                "boot_base_ci": np.percentile(boot_base, [2.5, 97.5]),
                "boot_spd_ci": np.percentile(boot_spd, [2.5, 97.5]),
                "p": p_value,
                "direction": "better" if ece_spd < ece_base else "worse",
            }

    # Plot
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

    domain_names = ["WGD", "LifeEval", "MedEval"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True, layout="constrained")
    x = np.arange(len(MODELS))
    bw = 0.38

    for ax, domain_name in zip(axes, domain_names):
        ece_bases = [results[(domain_name, m)]["ece_base"] for m in MODELS]
        ece_spds = [results[(domain_name, m)]["ece_spd"] for m in MODELS]
        colors = CB_PALETTE

        ax.bar(x - bw / 2, ece_bases, bw, color=colors,
               edgecolor="white", linewidth=0.5)
        ax.bar(x + bw / 2, ece_spds, bw, color=colors,
               edgecolor="white", linewidth=0.5, alpha=0.4)

        # Significance brackets
        for i, model in enumerate(MODELS):
            r = results[(domain_name, model)]
            if np.isnan(r["ece_base"]) or np.isnan(r["ece_spd"]):
                continue
            p = r["p"]
            label = sig_label(p)
            if label != "ns" and r["direction"] == "worse":
                label = label + "†"  # dagger for significantly worse
            bar_top = max(r["ece_base"], r["ece_spd"])
            y = bar_top + 0.008
            bracket_color = "#CC0000" if (label != "ns" and r["direction"] == "worse") else "#333333"
            ax.plot([i - bw / 2, i - bw / 2, i + bw / 2, i + bw / 2],
                    [y, y + 0.004, y + 0.004, y], color=bracket_color, linewidth=0.6)
            ax.text(i, y + 0.005, label, ha="center", va="bottom", fontsize=5.5,
                    color=bracket_color)

        ax.set_title(domain_name)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("ECE")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=1.0),
               plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.4)]
    axes[0].legend(handles, ["Baseline", "SPD"], loc="upper left")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "rq3_ece_significance.png"
    fig.savefig(out, dpi=300)
    print(f"\nSaved: {out}")

    # Print summary table
    print("\n--- Bootstrap significance summary (two-sided) ---")
    print(f"{'Domain':<12} {'Model':<28} {'ECE_base':>9} {'ECE_SPD':>9} {'p':>8} {'sig':>5} {'dir':>6}")
    for domain_name in domain_names:
        for model, label in zip(MODELS, MODEL_LABELS):
            r = results[(domain_name, model)]
            print(f"{domain_name:<12} {label:<28} {r['ece_base']:9.4f} {r['ece_spd']:9.4f} {r['p']:8.4f} {sig_label(r['p']):>5} {r['direction']:>6}")

    plt.close(fig)


if __name__ == "__main__":
    main()
