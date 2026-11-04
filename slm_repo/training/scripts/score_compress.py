from __future__ import annotations

import argparse
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from arena_grading import compress_judge, definition, generation_of, load_items, probe_of
from microtensor_subnet.validator import compress
from microtensor_subnet.validator.sandbox.judge import PROMPT

LINES = (0.5, 0.6, 0.7, 0.8, 0.9)
PROBES = {"asked": 0, "failed": 0}
LOCK = threading.Lock()


def ready(url: str, name: str) -> None:
    try:
        httpx.get(url.rstrip("/") + "/v1/models", timeout=30.0).raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"{name} at {url} is not answering yet ({exc}); start it and wait until it has loaded") from exc


def post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(url, json=body, timeout=900.0)
    response.raise_for_status()
    found: dict[str, Any] = response.json()
    return found


def complete(url: str, model: str, prompt: str, max_tokens: int, stop: list[str]) -> str:
    data = post(
        url.rstrip("/") + "/v1/completions",
        {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": max_tokens, "stop": stop},
    )
    return str((data.get("choices") or [{}])[0].get("text") or "")


def chat(url: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    with LOCK:
        PROBES["asked"] += 1
    try:
        data = post(
            url.rstrip("/") + "/v1/chat/completions",
            {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens},
        )
    except (httpx.HTTPError, ValueError):
        with LOCK:
            PROBES["failed"] += 1
        return ""
    return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="score a model on ctx_compress items the way validators do")
    parser.add_argument(
        "--items",
        required=True,
        help="items.jsonl from mta dev, a folder or file of the agent-sessions parquet, or microtensor-io/agent-sessions",
    )
    parser.add_argument("--arena", default="ctx_compress")
    parser.add_argument("--server", default="https://api.microtensor.cloud")
    parser.add_argument("--slm-url", required=True, help="your model behind an OpenAI compatible server")
    parser.add_argument("--slm-model", default="slm")
    parser.add_argument("--judge-url", required=True, help="the arena's probe model, see the line printed at start")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 uses the arena's generation.max_tokens, as validators do")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pass-line", type=float, default=None, help="defaults to the arena's pass threshold")
    parser.add_argument("--out", default="", help="write one scored line per item here")
    parser.add_argument("--summary", default="", help="write the totals as JSON here, for before_after.json")
    args = parser.parse_args()

    found = definition(args.server, args.arena)
    template = str(found.get("slm_prompt") or "")
    if not template:
        raise SystemExit(f"{args.arena} publishes no slm_prompt")
    max_tokens, stop = generation_of(found)
    max_tokens = args.max_tokens or max_tokens
    probe = probe_of(found)
    probe_tokens = int(probe.get("max_tokens") or 64)
    print(
        f"probe model {probe.get('model')}@{str(probe.get('revision') or '')[:8]} {probe.get('file') or ''}, "
        f"max_tokens {probe_tokens}, context {probe.get('max_input_tokens')}"
    )
    print(f"generation max_tokens {max_tokens}, stop {stop}")

    def answer(summary: str, questions: list[str]) -> list[str]:
        return [
            chat(
                args.judge_url,
                args.judge_model,
                [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"<summary>\n{summary}\n</summary>\n\nQuestion: {q}"},
                ],
                probe_tokens,
            )
            for q in questions
        ]

    judge = compress_judge(found, answer, args.pass_line)
    print(f"tokenizer {(found.get('grading') or {}).get('tokenizer')}, pass line {judge.pass_threshold}")
    ready(args.slm_url, "--slm-url")
    ready(args.judge_url, "--judge-url")
    items = [row for row in load_items(args.items) if not row.get("twin_of")][: args.limit]
    if not items:
        raise SystemExit(f"no items in {args.items}")

    def one(item: dict[str, Any]) -> dict[str, Any]:
        task = item["task"]
        label = item["label"]
        prompt = template.replace("{system}", task["system"]).replace("{user}", task["user"])
        try:
            output = complete(args.slm_url, args.slm_model, prompt, max_tokens, stop)
        except (httpx.HTTPError, ValueError) as exc:
            return {"item_id": item["item_id"], "score": 0.0, "reason": f"model call failed: {exc}", "parts": {}, "output": ""}
        verdict = compress.grade(output, label, task, judge)
        parts = dict(p.split(" ") for p in verdict.reason.split(", ")) if "recall" in verdict.reason else {}
        return {"item_id": item["item_id"], "score": verdict.score, "reason": verdict.reason, "parts": parts, "output": output}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, items))
    scores = [r["score"] for r in results]
    totals: dict[str, Any] = {
        "items": len(scores),
        "mean_item_score": round(statistics.mean(scores), 4),
        "pass_rate": round(sum(s >= judge.pass_threshold for s in scores) / len(scores), 4),
        "pass_line": judge.pass_threshold,
    }
    print(f"items {len(scores)}, mean item score {totals['mean_item_score']:.3f}")
    print(f"pass rate at the arena line {judge.pass_threshold:.2f}: {totals['pass_rate']:.1%}")
    for line in LINES:
        print(f"pass at {line:.1f}: {sum(s >= line for s in scores) / len(scores):.1%}")
    for key in ("recall", "invented", "continuation"):
        values = [float(r["parts"][key]) for r in results if key in r["parts"]]
        if values:
            totals[key] = round(statistics.mean(values), 4)
            print(f"{key}: mean {totals[key]:.3f}")
    totals["model_call_failures"] = sum(1 for r in results if r["reason"].startswith("model call failed"))
    totals["format_or_budget_failures"] = sum(1 for r in results if not r["parts"]) - totals["model_call_failures"]
    print(f"format or budget failures: {totals['format_or_budget_failures']}")
    print(f"model call failures: {totals['model_call_failures']}")
    totals["probe_failures"] = PROBES["failed"]
    print(f"probe failures: {PROBES['failed']} of {PROBES['asked']} probe questions")
    if PROBES["asked"] and PROBES["failed"] * 10 >= PROBES["asked"]:
        print("warning: the probe server failed on many questions, so continuation and pass rate are understated")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    if args.summary:
        Path(args.summary).write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
