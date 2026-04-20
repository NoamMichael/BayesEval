"""Run one (domain, model) task against a benchmark DataFrame."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .openrouter_client import OpenRouterClient


@dataclass
class RowResult:
    question_id: str
    answer: str | None
    confidence: float | None
    raw: str
    error: str | None
    tok_in: int = 0
    tok_out: int = 0


async def _run_one(
    client: OpenRouterClient,
    model: str,
    row: pd.Series,
    on_event,
    task_key: tuple[str, str],
) -> RowResult:
    image_path = row.get("image_path")
    image_uri = Path(image_path).read_text() if image_path else None
    resp = await client.complete(model, row["question_prompt"], row["confidence_prompt"],
                                 image_uri=image_uri)
    rr = RowResult(row["question_id"], resp.answer, resp.confidence, resp.raw, resp.error,
                   resp.tok_in, resp.tok_out)
    on_event(task_key, rr)
    return rr


async def run_task(
    client: OpenRouterClient,
    domain: str,
    model: str,
    benchmark: pd.DataFrame,
    mode: str,
    concurrency: int,
    on_event,
) -> pd.DataFrame:
    task_key = (domain, model)
    if mode == "seq":
        rows = [await _run_one(client, model, r, on_event, task_key) for _, r in benchmark.iterrows()]
    else:
        sem = asyncio.Semaphore(concurrency)

        async def guarded(r):
            async with sem:
                return await _run_one(client, model, r, on_event, task_key)

        rows = await asyncio.gather(*(guarded(r) for _, r in benchmark.iterrows()))
    return pd.DataFrame(
        [
            {
                "question_id": r.question_id,
                "Answer": r.answer,
                "Confidence": r.confidence,
                "raw": r.raw,
                "error": r.error,
            }
            for r in rows
        ]
    )
