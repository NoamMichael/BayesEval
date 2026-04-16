"""Async OpenRouter client. Chat Completions with JSON answer parsing."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

import httpx


@dataclass
class ModelResponse:
    answer: str | None
    confidence: float | None
    raw: str
    error: str | None


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse(raw: str) -> tuple[str | None, float | None, str | None]:
    if not raw:
        return None, None, "empty response"
    match = _JSON_RE.search(raw)
    if not match:
        return None, None, "no JSON object found"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, None, f"JSON decode: {e}"
    ans = obj.get("Answer")
    conf = obj.get("Confidence")
    try:
        conf_f = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        return ans, None, f"bad confidence: {conf!r}"
    return ans, conf_f, None


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str, timeout_s: float, max_retries: int):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, model: str, question: str, confidence_prompt: str) -> ModelResponse:
        messages = [
            {"role": "user", "content": question},
            {"role": "user", "content": confidence_prompt},
        ]
        payload = {"model": model, "messages": messages}
        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                r = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
                if r.status_code >= 500 or r.status_code == 429:
                    last_err = f"http {r.status_code}"
                    await asyncio.sleep(1.5 ** attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                raw = data["choices"][0]["message"]["content"]
                ans, conf, perr = _parse(raw)
                return ModelResponse(ans, conf, raw, perr)
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(1.5 ** attempt)
        return ModelResponse(None, None, "", last_err or "exhausted retries")
