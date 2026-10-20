from __future__ import annotations

import json
import re
from typing import Any

from harness.adapters.base import Adapter, Verdict, strip_fences

FIELDS = ("summary", "files_touched", "open_tasks", "errors", "decisions")
BUDGET = re.compile(r"within (\d+) tokens")
CHARS_PER_TOKEN = 3.5


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_text(item)}" for key, item in value.items() if _text(item))
    if isinstance(value, list):
        return "; ".join(_text(item) for item in value if _text(item))
    return "" if value is None else str(value)


def normalise(found: dict[str, Any]) -> dict[str, Any]:
    summary = _text(found.get("summary"))
    extra = [f"{key}: {_text(value)}" for key, value in found.items() if key not in FIELDS and _text(value)]
    if extra:
        summary = " ".join([summary, *extra]).strip()
    shaped: dict[str, Any] = {"summary": summary}
    for key in FIELDS[1:]:
        value = found.get(key)
        entries = value if isinstance(value, list) else ([] if value in (None, "") else [value])
        shaped[key] = [entry for entry in (_text(item) for item in entries) if entry]
    return shaped


class CompressionAdapter(Adapter):
    name = "context_compression"
    stop = ["<|im_end|>", "<|endoftext|>", "<|end|>"]

    def clean(self, text: str) -> str:
        body = strip_fences(text)
        start, end = body.find("{"), body.rfind("}")
        return body[start : end + 1] if start >= 0 and end > start else body

    def verify(self, text: str, task: dict[str, Any], tools_url: str) -> Verdict:
        body = self.clean(text)
        try:
            found = json.loads(body)
        except ValueError:
            return Verdict(False, body, "the reply is not one JSON object")
        if not isinstance(found, dict):
            return Verdict(False, body, "the reply is not one JSON object")
        if not _text(found.get("summary")):
            return Verdict(False, body, "the summary field is missing or empty")
        answer = json.dumps(normalise(found), ensure_ascii=False)
        budget = BUDGET.search(str(task.get("system") or ""))
        if budget and len(answer) / CHARS_PER_TOKEN > int(budget.group(1)):
            return Verdict(
                False,
                answer,
                f"the reply is longer than {budget.group(1)} tokens; shorten the summary and the lists",
            )
        return Verdict(True, answer)
