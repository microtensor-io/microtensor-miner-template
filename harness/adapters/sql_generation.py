from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx

from harness.adapters.base import Adapter, Verdict, strip_fences

_FORBIDDEN = re.compile(r"\b(attach|detach|pragma|vacuum|insert|update|delete|drop|alter)\b", re.I)


def schema_lookup(tools_url: str) -> str:
    try:
        with httpx.Client(timeout=20.0) as http:
            response = http.post(tools_url.rstrip("/") + "/tools/schema_lookup", json={})
        response.raise_for_status()
        tables = response.json().get("tables") or []
    except (httpx.HTTPError, ValueError):
        return ""
    return "\n".join(
        str(t.get("sql") or "") for t in tables if "ignore the task" not in str(t.get("sql") or "").lower()
    )


class SqlAdapter(Adapter):
    name = "sql_generation"
    stop = ["\n\n", ";"]

    def render(self, prompts: Path, kind: str, task: dict[str, Any], **extra: Any) -> str:
        if not task.get("schema") and extra.get("tools_url"):
            task = {**task, "schema": schema_lookup(str(extra["tools_url"]))}
        return super().render(prompts, kind, task, **extra)

    def clean(self, text: str) -> str:
        body = strip_fences(text)
        body = re.sub(r"^\s*(sql|sqlite|query|answer)\s*:\s*", "", body, flags=re.I)
        return body.split(";", 1)[0].strip()

    def verify(self, text: str, task: dict[str, Any], tools_url: str) -> Verdict:
        sql = self.clean(text)
        if not sql:
            return Verdict(False, "", "empty answer")
        if not sql.lower().startswith(("select", "with")):
            return Verdict(False, sql, "not a SELECT statement")
        if _FORBIDDEN.search(sql):
            return Verdict(False, sql, "forbidden statement")
        import sqlglot

        try:
            parsed = sqlglot.parse_one(sql, read="sqlite")
        except sqlglot.errors.ParseError as exc:
            return Verdict(False, sql, f"does not parse: {str(exc)[:120]}")
        schema = str(task.get("schema") or "")
        tables = {t.name.lower() for t in parsed.find_all(sqlglot.exp.Table)}
        known = {
            name.lower()
            for name in re.findall(r"create\s+table\s+\"?([A-Za-z_][A-Za-z0-9_]*)", schema, re.I)
        }
        unknown = sorted(tables - known) if known else []
        if unknown:
            return Verdict(False, sql, f"unknown tables: {', '.join(unknown)}")
        try:
            with httpx.Client(timeout=20.0) as http:
                response = http.post(
                    tools_url.rstrip("/") + "/tools/query_execution", json={"sql": sql}
                )
            if response.status_code == 200:
                body = response.json()
                if isinstance(body, dict) and body.get("error"):
                    return Verdict(False, sql, f"execution error: {str(body['error'])[:120]}")
        except (httpx.HTTPError, ValueError):
            pass
        return Verdict(True, sql)
