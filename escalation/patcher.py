from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from harness.adapters import Adapter

PROMPTS = Path(__file__).resolve().parent / "prompts"
MAX_TOKENS = 512
TRACE_LIMIT = 1200


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[-limit:]


def repair(
    task: dict[str, Any],
    draft: Any,
    trace: str,
    errors: str,
    *,
    adapter: Adapter,
    relay_url: str,
    ticket: str,
    model: str,
) -> Any:
    system = adapter.render(PROMPTS, "system", task)
    user = adapter.render(
        PROMPTS,
        "repair",
        task,
        draft=adapter.as_text(draft),
        errors=_trim(errors or "no verifier output", 400),
        trace=_trim(trace or "", TRACE_LIMIT),
    )
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    with httpx.Client(timeout=180.0) as http:
        response = http.post(
            relay_url.rstrip("/") + "/v1/chat/completions",
            json=body,
            headers={"x-mt-ticket": ticket},
        )
    if response.status_code != 200:
        return draft
    choices = response.json().get("choices") or [{}]
    text = str((choices[0].get("message") or {}).get("content") or "")
    fixed = adapter.clean(text)
    return fixed or draft
