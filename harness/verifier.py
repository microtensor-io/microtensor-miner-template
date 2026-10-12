from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
import sqlglot

_FENCE = re.compile(r"```[a-zA-Z]*")
_FORBIDDEN = re.compile(r"\b(attach|detach|pragma|vacuum|insert|update|delete|drop|alter)\b", re.I)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    sql: str
    reason: str = ""


def clean(text: str) -> str:
    body = _FENCE.sub("\n", text).strip()
    body = re.sub(r"^\s*(sql|sqlite|query|answer)\s*:\s*", "", body, flags=re.I)
    return body.split(";", 1)[0].strip()


def verify(text: str, schema: str, tools_url: str) -> Verdict:
    sql = clean(text)
    if not sql:
        return Verdict(False, "", "empty answer")
    if not sql.lower().startswith(("select", "with")):
        return Verdict(False, sql, "not a SELECT statement")
    if _FORBIDDEN.search(sql):
        return Verdict(False, sql, "forbidden statement")
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        return Verdict(False, sql, f"does not parse: {str(exc)[:120]}")
    tables = {t.name.lower() for t in parsed.find_all(sqlglot.exp.Table)}
    known = {name.lower() for name in re.findall(r"create\s+table\s+\"?([A-Za-z_][A-Za-z0-9_]*)", schema, re.I)}
    unknown = sorted(tables - known) if known else []
    if unknown:
        return Verdict(False, sql, f"unknown tables: {', '.join(unknown)}")
    try:
        with httpx.Client(timeout=20.0) as http:
            response = http.post(tools_url.rstrip("/") + "/tools/query_execution", json={"sql": sql})
        if response.status_code == 200:
            body = response.json()
            if isinstance(body, dict) and body.get("error"):
                return Verdict(False, sql, f"execution error: {str(body['error'])[:120]}")
    except (httpx.HTTPError, ValueError):
        pass
    return Verdict(True, sql)
