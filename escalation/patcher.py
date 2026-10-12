from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

PROMPTS = Path(__file__).resolve().parent / "prompts"
MAX_TOKENS = 256
TRACE_LIMIT = 1200


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[-limit:]


def _clean(text: str) -> str:
    body = re.sub(r"```[a-zA-Z]*", "\n", text).strip()
    body = re.sub(r"^\s*(sql|sqlite|query|answer)\s*:\s*", "", body, flags=re.I)
    return body.split(";", 1)[0].strip()


def repair(
    task: dict[str, Any],
    draft: Any,
    trace: str,
    errors: str,
    *,
    relay_url: str,
    ticket: str,
    model: str,
) -> Any:
    system = (PROMPTS / "system.txt").read_text(encoding="utf-8")
    user = (PROMPTS / "repair.txt").read_text(encoding="utf-8").format(
        schema=str(task.get("schema") or ""),
        question=str(task.get("question") or ""),
        draft=draft if isinstance(draft, str) else json.dumps(draft),
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
    fixed = _clean(text)
    return fixed or draft
