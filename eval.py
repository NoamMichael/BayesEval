#!/usr/bin/env python3
"""BayesEval cross-domain runner. Driven entirely by a YAML config.

Usage:
    python eval.py --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

from runner.dashboard import Dashboard
from runner.executor import run_task
from runner.openrouter_client import OpenRouterClient

REPO_ROOT = Path(__file__).resolve().parent
DOMAINS_DIR = REPO_ROOT / "domains"


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("domains") == "all" or cfg.get("domains") == ["all"]:
        cfg["domains"] = sorted(p.name for p in DOMAINS_DIR.iterdir() if p.is_dir())
    cfg.setdefault("mode", "batch")
    cfg.setdefault("concurrency", 8)
    cfg.setdefault("max_questions", None)
    cfg.setdefault("output_dir", "results")
    cfg.setdefault("openrouter", {})
    cfg["openrouter"].setdefault("api_key_env", "OPENROUTER_API_KEY")
    cfg["openrouter"].setdefault("base_url", "https://openrouter.ai/api/v1")
    cfg["openrouter"].setdefault("timeout_s", 60)
    cfg["openrouter"].setdefault("max_retries", 3)
    return cfg


def load_domain(name: str):
    """Return (benchmark_df, score_fn) for a domain."""
    domain_dir = DOMAINS_DIR / name
    benchmark_path = domain_dir / "Data" / "benchmark.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"{name}: missing benchmark at {benchmark_path}. "
            f"Run its build_benchmark.py first."
        )
    scoring_path = domain_dir / "scoring.py"
    if not scoring_path.exists():
        raise FileNotFoundError(f"{name}: missing scoring.py")
    spec = importlib.util.spec_from_file_location(f"{name}_scoring", scoring_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    benchmark = pd.read_csv(benchmark_path)
    return benchmark, module.score


def slugify(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


async def _main(cfg: dict) -> None:
    api_key = os.environ.get(cfg["openrouter"]["api_key_env"])
    if not api_key:
        sys.exit(f"Missing env var {cfg['openrouter']['api_key_env']}")

    output_dir = REPO_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_data: dict[str, tuple[pd.DataFrame, object]] = {}
    for d in cfg["domains"]:
        bench, score_fn = load_domain(d)
        if cfg["max_questions"]:
            bench = bench.head(cfg["max_questions"]).reset_index(drop=True)
        domain_data[d] = (bench, score_fn)

    client = OpenRouterClient(
        api_key=api_key,
        base_url=cfg["openrouter"]["base_url"],
        timeout_s=cfg["openrouter"]["timeout_s"],
        max_retries=cfg["openrouter"]["max_retries"],
    )

    title = (
        f"BayesEval — {len(cfg['models'])} model(s) × {len(cfg['domains'])} domain(s) "
        f"— mode: {cfg['mode']}"
    )
    summary_rows: list[dict] = []
    try:
        with Dashboard(title) as dash:
            for domain, (bench, _) in domain_data.items():
                for model in cfg["models"]:
                    dash.register(domain, model, total=len(bench))

            def on_event(key, row_result):
                dash.record(
                    key[0], key[1],
                    error=row_result.error is not None,
                    brier=None,
                )
                dash.refresh()

            for domain, (bench, score_fn) in domain_data.items():
                out_domain = output_dir / domain
                out_domain.mkdir(parents=True, exist_ok=True)
                tasks = [
                    run_task(client, domain, model, bench, cfg["mode"],
                             cfg["concurrency"], on_event)
                    for model in cfg["models"]
                ]
                results_list = await asyncio.gather(*tasks)
                for model, results in zip(cfg["models"], results_list):
                    out_path = out_domain / f"{slugify(model)}.csv"
                    results.to_csv(out_path, index=False)
                    scored = score_fn(results, bench)
                    mean_brier = float(scored["brier"].dropna().mean()) if "brier" in scored else float("nan")
                    err_count = int(results["error"].notna().sum())
                    summary_rows.append({
                        "domain": domain,
                        "model": model,
                        "n": len(results),
                        "errors": err_count,
                        "mean_brier": mean_brier,
                        "results_csv": str(out_path.relative_to(REPO_ROOT)),
                    })
            dash.refresh()
    finally:
        await client.aclose()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    asyncio.run(_main(cfg))


if __name__ == "__main__":
    main()
