from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from microtensor_subnet.validator import compress
from microtensor_subnet.validator.sandbox.judge import PROMPT

LINES = (0.5, 0.6, 0.7, 0.8, 0.9)
TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"


def post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(url, json=body, timeout=900.0)
    response.raise_for_status()
    found: dict[str, Any] = response.json()
    return found


def chat(url: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    data = post(
        url.rstrip("/") + "/v1/chat/completions",
        {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens},
    )
    return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="score a model on ctx_compress items the way validators do")
    parser.add_argument("--items", required=True, help="items.jsonl from mta dev")
    parser.add_argument("--slm-url", required=True, help="your model behind an OpenAI compatible server")
    parser.add_argument("--slm-model", default="slm")
    parser.add_argument("--judge-url", required=True, help="Qwen/Qwen2.5-Coder-7B-Instruct, the probe agent")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pass-line", type=float, default=0.5)
    parser.add_argument("--out", default="", help="write one scored line per item here")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(TOKENIZER)

    def answer(summary: str, questions: list[str]) -> list[str]:
        return [
            chat(
                args.judge_url,
                args.judge_model,
                [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"<summary>\n{summary}\n</summary>\n\nQuestion: {q}"},
                ],
                64,
            )
            for q in questions
        ]

    judge = compress.Judge(
        count_tokens=lambda text: len(tokenizer.encode(text).ids),
        answer=answer,
        pass_threshold=args.pass_line,
    )
    rows = [json.loads(line) for line in Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [row for row in rows if not row.get("twin_of")][: args.limit]

    def one(item: dict[str, Any]) -> dict[str, Any]:
        task = item["task"]
        output = chat(
            args.slm_url,
            args.slm_model,
            [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["user"]}],
            args.max_tokens,
        )
        verdict = compress.grade(output, item["label"], task, judge)
        parts = dict(p.split(" ") for p in verdict.reason.split(", ")) if "recall" in verdict.reason else {}
        return {"item_id": item["item_id"], "score": verdict.score, "reason": verdict.reason, "parts": parts, "output": output}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, items))
    scores = [r["score"] for r in results]
    print(f"items {len(scores)}, mean item score {statistics.mean(scores):.3f}")
    for line in LINES:
        print(f"pass at {line:.1f}: {sum(s >= line for s in scores) / len(scores):.1%}")
    for key in ("recall", "invented", "continuation"):
        values = [float(r["parts"][key]) for r in results if key in r["parts"]]
        if values:
            print(f"{key}: mean {statistics.mean(values):.3f}")
    failed = [r["reason"] for r in results if not r["parts"]]
    print(f"format or budget failures: {len(failed)}")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()
