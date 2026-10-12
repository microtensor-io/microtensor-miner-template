from __future__ import annotations

from typing import Any

from harness.adapters import Adapter, Verdict


def verify(adapter: Adapter, text: str, task: dict[str, Any], tools_url: str) -> Verdict:
    return adapter.verify(text, task, tools_url)
