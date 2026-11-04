from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from harness.adapters import Adapter

PROMPTS = Path(__file__).resolve().parent / "prompts"
MAX_TOKENS = 512
TRACE_LIMIT = 1200
RELAY_MAX_CHARS = 200_000
RELAY_MAX_TOKENS = 1024
RELAY_MARGIN = 4_000


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[-limit:]


def _prose(value: Any) -> bool:
    return isinstance(value, str) and any(mark in value[:4000] for mark in (" ", "\n"))


def _shorten(task: dict[str, Any], over: int) -> dict[str, Any]:
    fields = [k for k, v in task.items() if _prose(v)]
    if not fields or over <= 0:
        return task
    longest = max(fields, key=lambda k: len(task[k]))
    text = str(task[longest])
    keep = max(0, len(text) - over - 64)
    head = keep // 4
    tail = keep - head
    gap = f"\n[... {len(text) - keep} characters left out ...]\n"
    return {**task, longest: text[:head] + gap + (text[-tail:] if tail else "")}


def _messages(
    task: dict[str, Any], adapter: Adapter, draft: Any, trace: str, errors: str, limit: int = MAX_TOKENS
) -> list[dict[str, str]]:
    def build(source: dict[str, Any]) -> list[dict[str, str]]:
        system = adapter.render(PROMPTS, "system", source, limit=str(limit))
        user = adapter.render(
            PROMPTS,
            "repair",
            source,
            draft=adapter.as_text(draft),
            errors=_trim(errors or "no verifier output", 400),
            trace=_trim(trace or "", TRACE_LIMIT),
            limit=str(limit),
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    messages = build(task)
    size = sum(len(m["content"]) for m in messages)
    if size > RELAY_MAX_CHARS - RELAY_MARGIN:
        messages = build(_shorten(task, size - RELAY_MAX_CHARS + RELAY_MARGIN))
    return messages


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
    fitted: bool = False,
) -> Any:
    limit = min(adapter.repair_tokens(task, MAX_TOKENS), RELAY_MAX_TOKENS)
    body = {
        "model": model,
        "max_tokens": limit,
        "messages": _messages(task, adapter, draft, trace, errors, limit),
    }
    try:
        with httpx.Client(timeout=180.0) as http:
            response = http.post(
                relay_url.rstrip("/") + "/v1/chat/completions",
                json=body,
                headers={"x-mt-ticket": ticket},
            )
        if response.status_code != 200:
            return adapter.prefer(draft, "", task, fitted)
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return adapter.prefer(draft, "", task, fitted)
    choices = data.get("choices") if isinstance(data, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    text = str(message.get("content") or "") if isinstance(message, dict) else ""
    return adapter.prefer(draft, text, task, fitted)
