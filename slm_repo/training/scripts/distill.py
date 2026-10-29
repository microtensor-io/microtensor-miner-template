from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

FENCE = re.compile(r"^```[A-Za-z]*\s*|\s*```$")


def fetch_arena(server: str, arena: str) -> dict[str, Any]:
    response = httpx.get(f"{server.rstrip('/')}/v2/arenas/{arena}", timeout=60.0)
    response.raise_for_status()
    found: dict[str, Any] = response.json()
    return found


def chat(base_url: str, key: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    for attempt in range(4):
        try:
            response = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens},
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
            return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        except (httpx.HTTPError, ValueError):
            time.sleep(2.0 * (attempt + 1))
    return ""


def render(template: str, task: dict[str, Any]) -> str:
    text = template
    for key, value in task.items():
        if isinstance(value, str):
            text = text.replace("{" + key + "}", value)
    return text


def grader_for(arena: dict[str, Any]) -> Any:
    definition = arena.get("definition") or {}
    name = str(definition.get("grader") or "")
    if name == "context_compression":
        from microtensor_subnet.validator import compress

        grading = definition.get("grading") or {}
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(str(grading.get("tokenizer") or "Qwen/Qwen2.5-Coder-7B-Instruct"))
        judge = compress.Judge(
            count_tokens=lambda text: len(tokenizer.encode(text).ids),
            answer=lambda summary, questions: ["" for _ in questions],
            pass_threshold=float(grading.get("pass_threshold") or 0.5),
            min_output_tokens=int(grading.get("min_output_tokens") or 300),
            max_output_share=float(grading.get("max_output_share") or 0.1),
        )

        def grade(output: str, label: dict[str, Any], task: dict[str, Any]) -> tuple[bool, str]:
            verdict = compress.grade(output, label, task, judge)
            parts = dict(p.split(" ") for p in verdict.reason.split(", ")) if "recall" in verdict.reason else {}
            if not parts:
                return False, verdict.reason
            ok = float(parts.get("recall", 0)) >= 0.99 and float(parts.get("invented", 1)) == 0.0
            return ok, verdict.reason

        return grade

    from microtensor_subnet.validator import grade as grading_module

    def grade(output: str, label: dict[str, Any], task: dict[str, Any]) -> tuple[bool, str]:
        verdict = grading_module.grade(name, output, label, task)
        return bool(verdict.success), str(verdict.reason)

    return grade


def main() -> None:
    parser = argparse.ArgumentParser(description="write teacher answers for dev items and keep the ones the grader accepts")
    parser.add_argument("--arena", required=True)
    parser.add_argument("--items", required=True, help="items.jsonl built with the arena dev tooling, with task and label")
    parser.add_argument("--out", required=True, help="jsonl of {prompt, completion} for train.py")
    parser.add_argument("--teacher", required=True, help="teacher model id at your provider")
    parser.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--key-env", default="TEACHER_API_KEY")
    parser.add_argument("--server", default="https://api.microtensor.cloud")
    parser.add_argument("--max-tokens", type=int, default=0, help="default: the arena output budget")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-rejected", default="", help="write grader rejected answers here for inspection")
    args = parser.parse_args()

    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"set {args.key_env} to your provider key")
    arena = fetch_arena(args.server, args.arena)
    definition = arena.get("definition") or {}
    template = str(definition.get("slm_prompt") or "")
    if not template:
        raise SystemExit(f"{args.arena} publishes no slm_prompt")
    max_tokens = args.max_tokens or int((definition.get("generation") or {}).get("max_tokens") or 1024)
    grade = grader_for(arena)

    rows = [json.loads(line) for line in Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [row for row in rows if not row.get("twin_of")]
    if args.limit:
        items = items[: args.limit]

    def one(item: dict[str, Any]) -> dict[str, Any]:
        task = item["task"]
        messages = [{"role": "system", "content": str(task.get("system") or "")}]
        messages.append({"role": "user", "content": str(task.get("user") or json.dumps(task, sort_keys=True))})
        answer = FENCE.sub("", chat(args.base_url, key, args.teacher, messages, max_tokens).strip())
        ok, reason = grade(answer, item.get("label") or {}, task)
        return {
            "item_id": item.get("item_id"),
            "ok": ok,
            "reason": reason,
            "prompt": render(template, task),
            "completion": answer,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, items))
    kept = [r for r in results if r["ok"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n" for r in kept),
        encoding="utf-8",
    )
    if args.keep_rejected:
        Path(args.keep_rejected).write_text(
            "".join(json.dumps(r) + "\n" for r in results if not r["ok"]), encoding="utf-8"
        )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    record = Path(__file__).resolve().parents[1] / "distillation" / "teacher.json"
    record.write_text(
        json.dumps(
            {
                "teacher_model": args.teacher,
                "teacher_version": "",
                "provider": args.base_url,
                "items": len(results),
                "kept": len(kept),
                "pairs_sha256": digest,
                "harness_used": "none, the arena prompt",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(kept)} of {len(results)} teacher answers passed the grader; pairs at {out} (sha256 {digest[:12]})")


if __name__ == "__main__":
    main()
