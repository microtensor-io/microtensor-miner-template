from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FENCE = re.compile(r"```[a-zA-Z]*")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    answer: Any
    reason: str = ""


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def strip_fences(text: str) -> str:
    return _FENCE.sub("\n", text).strip()


def fields_of(task: dict[str, Any]) -> str:
    lines = []
    for key, value in task.items():
        body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {body}")
    return "\n".join(lines)


class Adapter:
    name = "generic"
    stop: list[str] = ["\n\n\n"]

    def render(self, prompts: Path, kind: str, task: dict[str, Any], **extra: Any) -> str:
        template = (prompts / self.name / f"{kind}.txt").read_text(encoding="utf-8")
        values = SafeDict(task_json=json.dumps(task, ensure_ascii=False), fields=fields_of(task))
        values.update({k: v for k, v in task.items() if isinstance(v, str | int | float)})
        values.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in extra.items()})
        return template.format_map(values)

    def query(self, task: dict[str, Any]) -> str:
        for key in ("question", "prompt", "input", "text", "instruction"):
            if isinstance(task.get(key), str):
                return str(task[key])
        return fields_of(task)

    def clean(self, text: str) -> str:
        return strip_fences(text)

    def verify(self, text: str, task: dict[str, Any], tools_url: str) -> Verdict:
        answer = self.clean(text)
        if not answer:
            return Verdict(False, "", "empty answer")
        return Verdict(True, answer)

    def as_text(self, answer: Any) -> str:
        return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
