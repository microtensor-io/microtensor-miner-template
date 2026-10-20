from __future__ import annotations

from harness.adapters.base import Adapter, Verdict
from harness.adapters.context_compression import CompressionAdapter
from harness.adapters.generic import GenericAdapter
from harness.adapters.sql_generation import SqlAdapter

ADAPTERS: dict[str, Adapter] = {
    GenericAdapter.name: GenericAdapter(),
    SqlAdapter.name: SqlAdapter(),
    CompressionAdapter.name: CompressionAdapter(),
}


def load(name: str) -> Adapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown adapter {name!r}; one of {sorted(ADAPTERS)}") from exc


__all__ = ["ADAPTERS", "Adapter", "Verdict", "load"]
