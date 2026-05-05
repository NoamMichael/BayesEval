#!/usr/bin/env python3
"""Bar chart of pathology frequencies in the MedEval dataset."""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "domains" / "MedEval" / "Data" / "benchmark_combined.csv"
FIG_DIR = ROOT / "analysis" / "figs" / "poster"


def main():
    df = pd.read_csv(BENCHMARK)
    # Use removal_pct=0 to avoid counting each patient 4x
    df = df[df["removal_pct"] == 0]
    counts = df["true_pathology"].value_counts()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    counts = counts.rename({
        "Bronchospasm / acute asthma exacerbation": "Bronchospasm",
        "Acute COPD exacerbation / infection": "COPD",
    })
    pcts = counts / counts.sum() * 100
    colors = cm.viridis(np.linspace(0.95, 0.05, len(pcts)))

    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    ax.bar(range(len(pcts)), pcts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels(pcts.index, rotation=45, ha="right")
    ax.set_ylabel("Frequency (%)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "medeval_pathologies.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
