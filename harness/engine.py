from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from harness.adapters import Adapter
from harness.verifier import verify

PROMPTS = Path(__file__).resolve().parent / "prompts"


@dataclass
class HarnessResult:
    answer: Any
    verified: bool
    errors: str = ""
    trace: str = ""
    steps: int = 0
    attempts: list[str] = field(default_factory=list)


def complete(slm_url: str, prompt: str, max_tokens: int, stop: list[str]) -> str:
    with httpx.Client(timeout=120.0) as http:
        response = http.post(
            slm_url.rstrip("/") + "/v1/completions",
            json={"model": "slm", "prompt": prompt, "max_tokens": max_tokens, "stop": stop},
        )
    response.raise_for_status()
    choices = response.json().get("choices") or [{}]
    return str(choices[0].get("text") or "").strip()


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
    errors = ""
    answer = ""
    trace: list[str] = []
    attempts: list[str] = []
    for step in range(1, max_steps + 1):
        kind = "task" if not errors else "retry"
        prompt = adapter.render(
            PROMPTS, kind, task, draft=answer, errors=errors, tools_url=tools_url
        )
        answer = complete(slm_url, prompt, max_tokens, adapter.stop)
        attempts.append(answer)
        verdict = verify(adapter, answer, task, tools_url)
        trace.append(f"step {step}: {'ok' if verdict.ok else verdict.reason}")
        if verdict.ok:
            return HarnessResult(
                answer=verdict.answer, verified=True, trace="\n".join(trace), steps=step, attempts=attempts
            )
        errors = verdict.reason
    return HarnessResult(
        answer=adapter.clean(answer),
        verified=False,
        errors=errors,
        trace="\n".join(trace),
        steps=max_steps,
        attempts=attempts,
    )
