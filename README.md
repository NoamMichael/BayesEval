# BayesEval

A benchmark suite for evaluating LLM calibration on Bayesian inference tasks across diverse domains.

## Overview

BayesEval measures whether language models can produce well-calibrated probabilistic reasoning — whether a model's stated confidence actually tracks its true accuracy. Unlike standard benchmarks with binary correct/incorrect answers, BayesEval uses **continuous ground-truth probabilities** derived from real-world distributions, scored with **strictly proper scoring rules** (Brier Score).

Each domain in BayesEval presents a different class of Bayesian inference problem with known analytical ground truth, enabling precise measurement of calibration quality.

## Domains

| Domain | Description | Ground Truth Source | Status |
|--------|-------------|-------------------|--------|
| [MedEval](domains/MedEval/) | Differential-diagnosis calibration | DDXPlus synthetic patient differentials | Scaffold |
| *More domains TBD* | | | |

## Scoring Framework

All domains share a common scoring framework based on strictly proper scoring rules:

- **Brier Score**: `BS = (confidence - true_probability)^2` — the primary metric
- **Murphy Decomposition**: `BS = Reliability - Resolution + Uncertainty`
  - **Reliability**: calibration quality (lower = better calibrated)
  - **Resolution**: sharpness of predictions (higher = more informative)
  - **Uncertainty**: intrinsic difficulty of the question

A perfectly calibrated model reports confidence `c` equal to the true probability `p` for each question, yielding `BS = 0`.

## Project Structure

```
BayesEval/
├── README.md
├── CLAUDE.md              # Development conventions and agent instructions
├── requirements.txt       # Python dependencies
├── thoughts/              # Research notes and experiment logs
└── domains/               # Domain-specific benchmark implementations (TBD)
```

## Getting Started

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...
```

## Running Evaluations

All runs are driven by `config.yaml` — edit it instead of passing flags:

```bash
python eval.py --config config.yaml
```

The config selects `domains` (list of folder names under `domains/`, or `all`),
`models` (OpenRouter slugs), `mode` (`batch` async or `seq`), and `concurrency`.
A live `rich` dashboard tracks progress per `(domain, model)` task. Raw
responses land in `results/<domain>/<model>.csv`; a combined `results/summary.csv`
is written at the end.

Domains are auto-discovered: drop a new folder under `domains/` with
`Data/benchmark.csv` and a `scoring.py` exposing `score(results_df, benchmark_df)`
and it will be picked up with no changes to `eval.py`.

## Related Work

This project is part of an honors thesis on LLM calibration. See individual domain READMEs for domain-specific methodology and results.
