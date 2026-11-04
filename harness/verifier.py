from __future__ import annotations

import inspect
from typing import Any

from harness.adapters import Adapter, Reply, Verdict


def _takes_reply(method: Any) -> bool:
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "reply" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in parameters)


def verify(
    adapter: Adapter, text: str, task: dict[str, Any], tools_url: str, reply: Reply | None = None
) -> Verdict:
    if reply is not None and _takes_reply(adapter.verify):
        return adapter.verify(text, task, tools_url, reply=reply)
    return adapter.verify(text, task, tools_url)
