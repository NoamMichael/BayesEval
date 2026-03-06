# BayesEval

A benchmark suite for evaluating LLM calibration on Bayesian inference tasks across diverse domains.

## Overview

BayesEval measures whether language models can produce well-calibrated probabilistic reasoning — whether a model's stated confidence actually tracks its true accuracy. Unlike standard benchmarks with binary correct/incorrect answers, BayesEval uses **continuous ground-truth probabilities** derived from real-world distributions, scored with **strictly proper scoring rules** (Brier Score).

Each domain in BayesEval presents a different class of Bayesian inference problem with known analytical ground truth, enabling precise measurement of calibration quality.

## Domains

| Domain | Description | Ground Truth Source | Status |
|--------|-------------|-------------------|--------|
| [LifeEval](../LifeEval/) | Actuarial mortality estimation | Gompertz distribution fitted to US life tables | Complete |
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
```

## Related Work

This project is part of an honors thesis on LLM calibration. See individual domain READMEs for domain-specific methodology and results.
