from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from harness.adapters.base import Adapter, Reply, Verdict, strip_fences

HEADINGS = (
    "Primary request and intent",
    "Key technical concepts",
    "Files and code sections",
    "Errors and fixes",
    "Problem solving",
    "All user messages",
    "Pending tasks",
    "Current work",
    "Next step",
)
BUDGET = re.compile(r"within (\d+) tokens")
LISTED = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
HASHES = re.compile(r"^#{1,6}\s*")
NUMBERED = re.compile(r"^\d{1,2}[.)]\s+")
BULLET = re.compile(r"^[-*+]\s")
HEADING_CHARS = 120
INJECTED = re.compile(r"^\s*SYSTEM NOTE:", re.IGNORECASE)
END_MARKS = ("<|end_of_text|>", "<|start_of_role|>", "<|im_end|>", "<|endoftext|>")
PIECES = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\w\r\n]|_)?[^\W\d_]+|\d{1,3}| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SAFETY = 1.12
SCALE_RANGE = (0.7, 1.5)
MIN_SAMPLE_CHARS = 400
EXACT_MARGIN = 4
EMPTY = "None."

Sections = dict[str, list[str]]


def headings_of(task: dict[str, Any]) -> tuple[str, ...]:
    found = tuple(h.strip() for h in LISTED.findall(str(task.get("system") or "")))
    return found if len(found) >= 3 else HEADINGS


def budget_of(task: dict[str, Any]) -> int:
    found = BUDGET.search(str(task.get("system") or ""))
    return int(found.group(1)) if found else 0


def generated(text: str) -> str:
    lines = text.rstrip().split("\n")
    while lines and INJECTED.match(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines)


def _body(text: str) -> str:
    body = strip_fences(generated(text))
    for mark in END_MARKS:
        body = body.split(mark, 1)[0]
    return body


def _bare(text: str) -> str:
    previous = None
    while text != previous:
        previous = text
        text = NUMBERED.sub("", HASHES.sub("", text.strip()))
        text = text.strip().strip("*_").strip().rstrip(":").strip()
    return text


def _heading(line: str, wanted: dict[str, str]) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped or BULLET.match(stripped):
        return "", ""
    name = wanted.get(_bare(stripped).casefold(), "") if len(stripped) <= HEADING_CHARS else ""
    if name:
        return name, ""
    if not (stripped.startswith(("#", "*", "_")) or NUMBERED.match(stripped)):
        return "", ""
    for mark in (":", " - "):
        head, found, rest = stripped.partition(mark)
        name = wanted.get(_bare(head).casefold(), "") if found and len(head) <= HEADING_CHARS else ""
        if name:
            return name, rest.strip().strip("*_").strip()
    return "", ""


def parse(text: str, headings: Sequence[str]) -> tuple[Sections, list[str]]:
    wanted = {h.casefold(): h for h in headings}
    sections: Sections = {}
    current = ""
    for line in _body(text).splitlines():
        name, rest = _heading(line, wanted)
        if name:
            current = name
            sections.setdefault(name, [])
            if rest:
                sections[name].append(rest)
        elif current:
            sections[current].append(line.rstrip())
    for lines in sections.values():
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
    return sections, [h for h in headings if h not in sections]


def empty_sections(sections: Sections, headings: Sequence[str]) -> list[str]:
    return [
        h for h in headings if not any(line.strip() and line.strip() != EMPTY for line in sections.get(h) or [])
    ]


def render(sections: Sections, headings: Sequence[str]) -> str:
    parts = []
    for name in headings:
        lines = sections.get(name) or []
        parts.append(f"## {name}\n" + ("\n".join(lines) if lines else EMPTY))
    return "\n\n".join(parts) + "\n"


def estimate(text: str) -> float:
    total = 0.0
    for piece in PIECES.findall(text):
        if not piece.strip() or piece.isdigit():
            total += 1.0
        elif not piece.isascii():
            total += max(1.0, len(piece.encode("utf-8")) / 2)
        elif piece[-1].isalpha():
            total += 1.0 + max(0, len(piece) - 6) * 0.25
        else:
            total += 1.0 if len(piece) <= 3 else len(piece) * 0.5
    return total


def scale_of(reply: Reply | None) -> float:
    if reply is None or reply.cut_short or reply.tokens <= 0:
        return SAFETY
    sample = generated(reply.text)
    guess = estimate(sample)
    if len(sample) < MIN_SAMPLE_CHARS or guess <= 0:
        return SAFETY
    low, high = SCALE_RANGE
    return min(max(reply.tokens / guess, low), high)


def _shorten(line: str, keep: float) -> str:
    prefix = line[: max(0, int(len(line) * keep))]
    return prefix.rsplit(" ", 1)[0] if " " in prefix.strip() else ""


def fit(sections: Sections, headings: Sequence[str], budget: int, scale: float, exact: int = 0) -> str:
    text = render(sections, headings)
    if not budget or (exact and exact <= budget - EXACT_MARGIN):
        return text
    limit = budget - max(32, budget // 40)
    sections = {name: list(sections.get(name) or []) for name in headings}
    total = estimate(text) * scale
    while total > limit:
        costs = {name: [estimate(line + "\n") * scale for line in lines] for name, lines in sections.items()}
        running = total
        while running > limit:
            name = max(headings, key=lambda h: sum(costs[h]))
            lines = sections[name]
            if not lines:
                break
            if len(lines) > 1:
                lines.pop()
                running -= costs[name].pop()
                while lines and not lines[-1].strip():
                    lines.pop()
                    running -= costs[name].pop()
            else:
                cost = costs[name][0]
                keep = max(0.0, 1 - (running - limit) / cost - 0.02) if cost else 0.0
                kept = _shorten(lines[0], keep)
                sections[name] = [kept] if kept else []
                costs[name] = [estimate(kept + "\n") * scale] if kept else []
                running -= cost - sum(costs[name])
        text = render(sections, headings)
        updated = estimate(text) * scale
        if updated >= total:
            break
        total = updated
    return text


def exact_tokens(reply: Reply | None, shaped: str) -> int:
    if reply is None or reply.cut_short or reply.tokens <= 0:
        return 0
    return reply.tokens if shaped.strip() == reply.text.strip() else 0


def shaped(text: str, task: dict[str, Any], reply: Reply | None = None) -> tuple[str, list[str]]:
    headings = headings_of(task)
    sections, missing = parse(text, headings)
    if len(missing) == len(headings):
        body = [line.rstrip() for line in _body(text).strip().splitlines()]
        sections = {headings[0]: body} if any(line.strip() for line in body) else {}
    exact = exact_tokens(reply, render(sections, headings))
    return fit(sections, headings, budget_of(task), scale_of(reply), exact), missing


class CompressionAdapter(Adapter):
    name = "context_compression"
    stop = ["<|end_of_text|>", "<|start_of_role|>"]

    def max_tokens(self, task: dict[str, Any], default: int) -> int:
        budget = budget_of(task)
        return min(default, budget) if budget else default

    def repair_tokens(self, task: dict[str, Any], default: int) -> int:
        return budget_of(task) or default

    def clean(self, text: str) -> str:
        sections, _ = parse(text, HEADINGS)
        return render(sections, HEADINGS)

    def draft(self, text: str) -> str:
        return _body(text).strip()

    def verify(
        self, text: str, task: dict[str, Any], tools_url: str, reply: Reply | None = None
    ) -> Verdict:
        headings = headings_of(task)
        answer, missing = shaped(text, task, reply)
        cut = reply is not None and reply.finish == "length"
        if len(missing) == len(headings):
            return Verdict(
                False,
                answer,
                f"none of the {len(headings)} headings came back; write each on its own line as '## Heading'",
            )
        if missing:
            why = "it ran past the token budget and was cut off" if cut else "headings are missing"
            return Verdict(
                False,
                answer,
                f"{why}: {', '.join(missing)}; write all {len(headings)} headings in order, shorter",
            )
        return Verdict(True, answer)

    def settle(self, replies: Sequence[Reply], task: dict[str, Any]) -> Any:
        headings = headings_of(task)
        best: tuple[int, Reply] | None = None
        for reply in replies:
            if not reply.text:
                continue
            _, missing = parse(reply.text, headings)
            if best is None or len(missing) <= best[0]:
                best = (len(missing), reply)
        if best is None:
            return render({}, headings)
        return shaped(best[1].text, task, best[1])[0]

    def prefer(self, draft: Any, text: str, task: dict[str, Any], fitted: bool = False) -> Any:
        headings = headings_of(task)
        previous = self.as_text(draft) if draft else ""
        after = empty_sections(parse(text, headings)[0], headings)
        before = empty_sections(parse(previous, headings)[0], headings)
        if previous.strip() and (not text.strip() or len(after) > len(before)):
            return draft if fitted else shaped(previous, task)[0]
        return shaped(text, task)[0]
