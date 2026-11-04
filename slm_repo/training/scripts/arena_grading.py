from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

Answer = Callable[[str, Sequence[str]], list[str]]
TASK_FIELDS = ("system", "user")
ROW_FIELDS = ("item_id", "twin_of", "family")
JSON_FIELDS = ("facts", "probes")


def definition(server: str, arena: str) -> dict[str, Any]:
    response = httpx.get(f"{server.rstrip('/')}/v2/arenas/{arena}", timeout=60.0)
    response.raise_for_status()
    found = response.json().get("definition") or {}
    if not isinstance(found, dict) or not found:
        raise SystemExit(f"{arena} publishes no definition")
    return found


def generation_of(found: dict[str, Any]) -> tuple[int, list[str]]:
    generation = found.get("generation") or {}
    if not generation.get("max_tokens"):
        raise SystemExit("the arena definition has no generation.max_tokens")
    return int(generation["max_tokens"]), [str(s) for s in generation.get("stop") or []]


def token_counter(name: str) -> Callable[[str], int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(name)
    return lambda text: len(tokenizer.encode(text).ids)


def compress_judge(found: dict[str, Any], answer: Answer, pass_line: float | None = None) -> Any:
    from microtensor_subnet.validator import compress

    grading = found.get("grading") or {}
    name = str(grading.get("tokenizer") or "")
    if not name:
        raise SystemExit("the arena definition names no grading tokenizer")
    return compress.Judge(
        count_tokens=token_counter(name),
        answer=answer,
        pass_threshold=float(pass_line if pass_line is not None else grading["pass_threshold"]),
        min_output_tokens=int(grading["min_output_tokens"]),
        max_output_share=float(grading["max_output_share"]),
        max_output_tokens=int(grading["max_output_tokens"]),
        headings=tuple(str(h) for h in grading.get("headings") or compress.HEADINGS),
    )


def probe_of(found: dict[str, Any]) -> dict[str, Any]:
    return dict((found.get("evaluation") or {}).get("probe") or {})


def _item(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("task"), dict):
        return row
    label = {k: v for k, v in row.items() if k not in TASK_FIELDS and k not in ROW_FIELDS}
    for key in JSON_FIELDS:
        if isinstance(label.get(key), str):
            label[key] = json.loads(label[key])
    return {
        "item_id": str(row.get("item_id") or ""),
        "twin_of": str(row.get("twin_of") or ""),
        "family": str(row.get("family") or ""),
        "task": {key: str(row.get(key) or "") for key in TASK_FIELDS},
        "label": label,
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_table(path).to_pylist())
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_items(source: str) -> list[dict[str, Any]]:
    path = Path(source)
    if path.is_dir():
        files = sorted(path.glob("*.parquet")) or sorted(path.glob("items.jsonl"))
        if not files:
            raise SystemExit(f"{source} holds no .parquet files and no items.jsonl")
    elif path.is_file():
        files = [path]
    else:
        from datasets import load_dataset

        name, _, split = source.partition(":")
        return [_item(dict(row)) for row in load_dataset(name, split=split or "dev")]
    return [_item(row) for file in files for row in _rows(file)]
