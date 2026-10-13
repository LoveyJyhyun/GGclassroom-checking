"""Check numeric answers against a teacher-provided answer key.

The LLM (optional) only *extracts* each student's final answers; all
comparisons are done here in Python, because small models are unreliable
at arithmetic.
"""
from __future__ import annotations

import math
import re

from . import llm

_NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?%?")
_LABEL = re.compile(
    r"^\s*(?:q(?:uestion)?|câu|bài|ex(?:ercise)?|problem|#)?\s*(\d+[a-z]?)\s*[:.)=\-]\s*(.*)$",
    re.IGNORECASE,
)
# Tolerances may use a decimal comma ("± 0,01") and may be written out in words
# or in brackets, which is how Vietnamese answer sheets usually put it:
#   "3,14 ± 0,01"   "3.14 (sai số 0,01)"   "3.14 (sai so 0.01)"   "1/3 +- 1%"
_TOL = re.compile(r"[(\[]?\s*(?:±|\+/-|\+-|sai\s*s[oố]|tolerance|tol)\s*[:=]?\s*([\d.,]*\d\s*%?)\s*[)\]]?\s*$", re.IGNORECASE)


def parse_number(token: str) -> float | None:
    """Parse '1,234.5', '3,14', '-2.5e3', '3/4', '42%' style numbers."""
    t = token.strip().replace("−", "-").replace(" ", "")
    percent = t.endswith("%")
    t = t.rstrip("%")
    try:
        if "/" in t:
            num, den = t.split("/", 1)
            value = parse_number(num) / parse_number(den)
        else:
            if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", t):
                t = t.replace(",", "")          # thousands separators
            elif t.count(",") == 1 and "." not in t:
                t = t.replace(",", ".")         # decimal comma
            value = float(t)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value / 100 if percent else value


def last_number(text: str) -> float | None:
    matches = _NUM.findall(text or "")
    for m in reversed(matches):
        value = parse_number(m)
        if value is not None:
            return value
    return None


def parse_key(text: str) -> dict[str, dict]:
    """Parse an answer key: one 'label: value [± tol]' per line.

    Examples:  "1: 42"   "2) 3.14 ± 0.01"   "3a = 25%"   "4: 1/3 ± 1%"
    """
    key = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LABEL.match(line)
        if not m:
            continue
        label, rest = m.group(1).lower(), m.group(2)
        tol = None
        tm = _TOL.search(rest)
        if tm:
            tol = tm.group(1)
            rest = rest[:tm.start()]
        value = last_number(rest)
        if value is not None:
            key[label] = {"value": value, "tol": tol, "raw": rest.strip()}
    return key


def is_close(expected: float, got: float, tol: str | None = None, default_rel: float = 1e-3) -> bool:
    if tol:
        tol = tol.strip().replace(" ", "")
        amount = parse_number(tol.rstrip("%"))
        if amount is not None:                      # an unreadable tolerance falls through
            if tol.endswith("%"):
                return abs(got - expected) <= abs(expected) * amount / 100
            return abs(got - expected) <= amount
    return math.isclose(got, expected, rel_tol=default_rel, abs_tol=1e-9)


def extract_by_labels(text: str) -> dict[str, float]:
    """Find 'label: ... number' lines in the student's text."""
    found = {}
    for line in (text or "").splitlines():
        m = _LABEL.match(line)
        if m:
            value = last_number(m.group(2))
            if value is not None:
                found[m.group(1).lower()] = value  # later lines win (final answers)
    return found


EXTRACT_SYSTEM = (
    "You extract the FINAL numeric answers a student gave for each question. "
    "Do not solve the problems and do not correct the student. "
    'Reply with JSON only: {"answers": {"<question label>": "<number as written by the student or null>"}}'
)


def extract_with_llm(text: str, labels: list[str]) -> tuple[dict[str, float], list[str]]:
    """Return (label -> number the student wrote, sections that had to be trimmed)."""
    data, trimmed = llm.pack_json(EXTRACT_SYSTEM, [("Question labels", ", ".join(labels)),
                                                   ("Student submission", text)])
    found = {}
    for label, raw in (data.get("answers") or {}).items():
        value = parse_number(str(raw)) if raw is not None else None
        if value is None and raw is not None:
            value = last_number(str(raw))
        if value is not None:
            found[str(label).lower().strip()] = value
    return found, trimmed


def check(text: str, key: dict[str, dict], use_llm: bool = True,
          default_rel: float = 1e-3) -> dict:
    found = extract_by_labels(text)
    source = "pattern"
    missing = [label for label in key if label not in found]
    llm_error, truncated = None, []
    if missing and use_llm and text.strip():
        try:
            extra, truncated = extract_with_llm(text, missing)
            for label in missing:
                if label in extra:
                    found[label] = extra[label]
            source = "pattern + llm"
        except llm.LLMError as e:
            llm_error = str(e)

    results, correct = [], 0
    for label, exp in key.items():
        got = found.get(label)
        ok = got is not None and is_close(exp["value"], got, exp["tol"], default_rel)
        correct += ok
        results.append({"question": label, "expected": exp["value"], "tol": exp["tol"],
                        "got": got, "ok": ok})
    return {"results": results, "correct": correct, "total": len(key),
            "source": source, "error": llm_error, "truncated": truncated}
