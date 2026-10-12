from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from harness.verifier import verify

PROMPTS = Path(__file__).resolve().parent / "prompts"
STOP = ["\n\n", ";"]


@dataclass
class HarnessResult:
    answer: Any
    verified: bool
    errors: str = ""
    trace: str = ""
    steps: int = 0
    attempts: list[str] = field(default_factory=list)


def _prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def complete(slm_url: str, prompt: str, max_tokens: int) -> str:
    with httpx.Client(timeout=120.0) as http:
        response = http.post(
            slm_url.rstrip("/") + "/v1/completions",
            json={"model": "slm", "prompt": prompt, "max_tokens": max_tokens, "stop": STOP},
        )
    response.raise_for_status()
    choices = response.json().get("choices") or [{}]
    return str(choices[0].get("text") or "").strip()


def schema_lookup(tools_url: str) -> str:
    try:
        with httpx.Client(timeout=20.0) as http:
            response = http.post(tools_url.rstrip("/") + "/tools/schema_lookup", json={})
        response.raise_for_status()
        tables = response.json().get("tables") or []
    except (httpx.HTTPError, ValueError):
        return ""
    lines = []
    for table in tables:
        sql = str(table.get("sql") or "")
        if "ignore the task" in sql.lower():
            continue
        lines.append(sql)
    return "\n".join(lines)


def run_harness(task: dict[str, Any], *, slm_url: str, tools_url: str, config: dict[str, Any]) -> HarnessResult:
    schema = str(task.get("schema") or "")
    question = str(task.get("question") or "")
    max_tokens = int((config.get("slm") or {}).get("max_new_tokens") or 256)
    max_steps = int((config.get("harness") or {}).get("max_steps") or 3)
    live_schema = schema_lookup(tools_url)
    prompt = _prompt("task.txt").format(schema=schema or live_schema, question=question)
    errors = ""
    trace: list[str] = []
    answer = ""
    for step in range(1, max_steps + 1):
        attempt = prompt if not errors else _prompt("retry.txt").format(
            schema=schema or live_schema, question=question, draft=answer, errors=errors
        )
        answer = complete(slm_url, attempt, max_tokens)
        verdict = verify(answer, schema, tools_url)
        trace.append(f"step {step}: {'ok' if verdict.ok else verdict.reason}")
        if verdict.ok:
            return HarnessResult(answer=verdict.sql, verified=True, trace="\n".join(trace), steps=step)
        errors = verdict.reason
    return HarnessResult(answer=answer, verified=False, errors=errors, trace="\n".join(trace), steps=max_steps)
