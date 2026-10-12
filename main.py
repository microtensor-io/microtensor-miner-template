from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from escalation.patcher import repair  # noqa: E402
from harness.engine import run_harness  # noqa: E402

CONFIG: dict[str, Any] = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SLM_URL = os.environ["MT_SLM_URL"]
ROUTER_URL = os.environ.get("MT_ROUTER_URL", "")
TOOLS_URL = os.environ["MT_TOOLS_URL"]
RELAY_URL = os.environ["MT_RELAY_URL"]
PREFERRED_MODEL = str((CONFIG.get("escalation") or {}).get("preferred_model") or "")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class HarnessIn(BaseModel):
    item_id: str
    task: dict[str, Any]


class RepairIn(BaseModel):
    item_id: str
    task: dict[str, Any]
    draft: Any = None
    trace: str = ""
    errors: str = ""
    ticket: str = ""


class SystemIn(BaseModel):
    item_id: str
    task: dict[str, Any]
    floor: float = Field(ge=0.0, le=1.0)
    ticket: str = ""


def decide(query: str, draft: str) -> float:
    if not ROUTER_URL:
        return 1.0
    with httpx.Client(timeout=30.0) as http:
        response = http.post(
            ROUTER_URL.rstrip("/") + "/decide", json={"input": {"query": query, "draft": draft}}
        )
    response.raise_for_status()
    answers = response.json().get("answers") or []
    return float(answers[0]["probability"]) if answers else 0.0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/harness/run")
def harness_run(payload: HarnessIn) -> dict[str, Any]:
    result = run_harness(payload.task, slm_url=SLM_URL, tools_url=TOOLS_URL, config=CONFIG)
    return {
        "answer": result.answer,
        "self_check": {"pass": result.verified},
        "trace": result.trace,
        "steps": result.steps,
    }


@app.post("/escalation/repair")
def escalation_repair(payload: RepairIn) -> dict[str, Any]:
    answer = repair(
        payload.task,
        payload.draft,
        payload.trace,
        payload.errors,
        relay_url=RELAY_URL,
        ticket=payload.ticket,
        model=PREFERRED_MODEL,
    )
    return {"answer": answer}


@app.post("/system/run")
def system_run(payload: SystemIn) -> dict[str, Any]:
    result = run_harness(payload.task, slm_url=SLM_URL, tools_url=TOOLS_URL, config=CONFIG)
    draft = result.answer
    query = str(payload.task.get("question", ""))
    probability = decide(query, json.dumps(draft) if not isinstance(draft, str) else draft)
    if result.verified and probability >= payload.floor:
        return {"answer": draft, "route": "local", "p": probability}
    answer = repair(
        payload.task,
        draft,
        result.trace,
        result.errors,
        relay_url=RELAY_URL,
        ticket=payload.ticket,
        model=PREFERRED_MODEL,
    )
    return {"answer": answer, "route": "escalated", "p": probability}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MT_PORT", "8080")), log_level="warning")
