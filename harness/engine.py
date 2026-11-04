from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from harness.adapters import Adapter, Reply
from harness.verifier import verify

PROMPTS = Path(__file__).resolve().parent / "prompts"
REFUSED = 429


@dataclass
class HarnessResult:
    answer: Any
    verified: bool
    errors: str = ""
    trace: str = ""
    steps: int = 0
    attempts: list[str] = field(default_factory=list)


def complete(slm_url: str, prompt: str, max_tokens: int, stop: list[str]) -> Reply:
    try:
        with httpx.Client(timeout=120.0) as http:
            response = http.post(
                slm_url.rstrip("/") + "/v1/completions",
                json={"model": "slm", "prompt": prompt, "max_tokens": max_tokens, "stop": stop},
            )
    except httpx.HTTPError as exc:
        return Reply("", failed=f"the model call failed ({type(exc).__name__})", cap=max_tokens)
    if response.status_code == REFUSED:
        return Reply("", failed="the model call limit for this item is reached", cap=max_tokens, final=True)
    if response.status_code != 200:
        return Reply("", failed=f"the model call failed with status {response.status_code}", cap=max_tokens)
    try:
        data = response.json()
    except ValueError:
        return Reply("", failed="the model reply was not valid JSON", cap=max_tokens)
    if not isinstance(data, dict):
        return Reply("", failed="the model reply was not an object", cap=max_tokens)
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tokens = usage.get("completion_tokens")
    return Reply(
        text=str(choice.get("text") or "").strip(),
        finish=str(choice.get("finish_reason") or ""),
        tokens=tokens if isinstance(tokens, int) and tokens > 0 else 0,
        cap=max_tokens,
    )


def run_harness(
    task: dict[str, Any],
    *,
    adapter: Adapter,
    slm_url: str,
    tools_url: str,
    config: dict[str, Any],
) -> HarnessResult:
    max_tokens = int((config.get("slm") or {}).get("max_new_tokens") or 256)
    max_steps = int((config.get("harness") or {}).get("max_steps") or 3)
    kind = "task"
    draft = ""
    errors = ""
    trace: list[str] = []
    replies: list[Reply] = []
    for step in range(1, max_steps + 1):
        prompt = adapter.render(PROMPTS, kind, task, draft=draft, errors=errors, tools_url=tools_url)
        reply = complete(slm_url, prompt, adapter.max_tokens(task, max_tokens), adapter.stop)
        replies.append(reply)
        if reply.failed or reply.cut_short:
            errors = reply.failed or "the reply was cut off before the model finished"
            trace.append(f"step {step}: {errors}")
            if reply.final:
                break
            kind = "task"
            continue
        verdict = verify(adapter, reply.text, task, tools_url, reply)
        trace.append(f"step {step}: {'ok' if verdict.ok else verdict.reason}")
        if verdict.ok:
            return HarnessResult(
                answer=verdict.answer,
                verified=True,
                trace="\n".join(trace),
                steps=step,
                attempts=[r.text for r in replies],
            )
        errors = verdict.reason
        draft = adapter.draft(reply.text)
        kind = verdict.next
        if not kind:
            break
    return HarnessResult(
        answer=adapter.settle(replies, task),
        verified=False,
        errors=errors,
        trace="\n".join(trace),
        steps=len(replies),
        attempts=[r.text for r in replies],
    )
