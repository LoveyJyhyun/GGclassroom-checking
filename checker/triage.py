"""Turn a check report into a prioritised follow-up list for the teacher.

Everything in this module is deterministic. The agent (a Dify workflow on a local
Qwen) only writes the narrative on top of it and suggests whose grade to hold, so
the facts it works from are the same every time and come straight from the report.

The agent never sees names, emails or answers. Each student becomes a stable code
(S01, S02, ... in name order), and free text that came out of an earlier LLM call
(copy evidence, review notes) is scrubbed of names and emails before it leaves.
Even an agent wired to a cloud model by mistake would learn nothing about who the
students are.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from statistics import mean, median

from . import results
from . import status as st

ACT, REVIEW, OK = "act", "review", "ok"
TIER_LABELS = {ACT: "Needs action", REVIEW: "Worth a look", OK: "No concerns"}
HIGH, MEDIUM, INFO = "high", "medium", "info"

# Copying. Shared 5-word sequences are strong evidence; meaning similarity alone is
# weak (every answer to the same question is "about the same thing"), and identical
# code is normal for tiny programs, so those two only ask for a look.
COPY_LEXICAL_HIGH = 0.5
COPY_LEXICAL_MEDIUM = 0.25
COPY_CODE_MEDIUM = 0.9
COPY_SEMANTIC_MEDIUM = 0.97
COPY_LLM_CONFIDENT = 70
# AI-writing indicators are never proof, so they never reach "high" on their own.
AI_HEURISTIC_MEDIUM, AI_HEURISTIC_INFO = 60, 40
AI_LLM_MEDIUM, AI_LLM_INFO = 70, 50
LOW_SCORE = 50.0
EVIDENCE_CHARS = 160

CODE_RE = re.compile(r"\bS\d{2,3}\b")


# --- Codes and scrubbing -----------------------------------------------------

def codes(report: dict) -> dict[str, str]:
    """uid -> S01, S02, ... in name order, stable for a given report."""
    students = report.get("students") or {}
    order = sorted(students, key=lambda uid: ((students[uid].get("name") or "").lower(), uid))
    width = 3 if len(order) > 99 else 2
    return {uid: f"S{i:0{width}d}" for i, uid in enumerate(order, 1)}


def _scrubber(report: dict, code_of: dict[str, str]):
    """A function that replaces every student's name and email with their code."""
    pairs = []
    for uid, s in (report.get("students") or {}).items():
        for token in {s.get("name") or "", s.get("email") or "",
                      (s.get("email") or "").split("@")[0]}:
            if len(token) >= 3:
                pairs.append((token, code_of[uid]))
    pairs.sort(key=lambda p: -len(p[0]))         # "Nguyen Van An" before "An..."
    patterns = [(re.compile(re.escape(t), re.IGNORECASE), c) for t, c in pairs]

    def scrub(text) -> str:
        text = " ".join(str(text or "").split())
        for pattern, code in patterns:
            text = pattern.sub(code, text)
        return text
    return scrub


def _clip(text: str, limit: int = EVIDENCE_CHARS) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# --- Signals -----------------------------------------------------------------

def _signal(level: str, kind: str, text: str) -> dict:
    return {"level": level, "kind": kind, "text": text}


def _copy_signals(report: dict, uid: str, code_of: dict, scrub) -> list[dict]:
    out = []
    for p in report.get("duplicates") or []:
        if uid not in (p["a"], p["b"]):
            continue
        other = code_of.get(p["b"] if p["a"] == uid else p["a"], "?")
        lexical, semantic, code = p.get("lexical") or 0, p.get("semantic") or 0, p.get("code") or 0
        review = p.get("llm") or {}
        confidence = review.get("confidence") or 0

        level = None
        if (review.get("copied") and confidence >= COPY_LLM_CONFIDENT) or lexical >= COPY_LEXICAL_HIGH:
            level = HIGH
        elif (lexical >= COPY_LEXICAL_MEDIUM or code >= COPY_CODE_MEDIUM
              or semantic >= COPY_SEMANTIC_MEDIUM or review.get("copied")):
            level = MEDIUM
        if review and not review.get("copied") and confidence >= COPY_LLM_CONFIDENT:
            level = INFO                               # the reviewer looked and cleared it
        if level is None:
            continue

        parts = [f"{name} {value:.0%}" for name, value in
                 (("wording", p.get("lexical")), ("meaning", p.get("semantic")), ("code", p.get("code")))
                 if value is not None]
        text = f"similar to {other}: " + ", ".join(parts)
        if review:
            verdict = "likely copied" if review.get("copied") else "probably not copied"
            text += f"; reviewer: {verdict} ({confidence}%)"
            evidence = [scrub(e) for e in review.get("evidence") or [] if str(e).strip()]
            if evidence:
                text += f' - "{_clip(evidence[0])}"'
        out.append(_signal(level, "copying", text))
    return out


def _ai_signal(report: dict, uid: str, scrub) -> dict | None:
    r = (report.get("ai") or {}).get(uid)
    if not r:
        return None
    h = (r.get("heuristic") or {}).get("score")
    llm_view = r.get("llm") or {}
    likelihood = llm_view.get("likelihood")
    if (h or 0) >= AI_HEURISTIC_MEDIUM or (likelihood or 0) >= AI_LLM_MEDIUM:
        level = MEDIUM
    elif (h or 0) >= AI_HEURISTIC_INFO or (likelihood or 0) >= AI_LLM_INFO:
        level = INFO
    else:
        return None
    text = (f"AI-writing indicators (not proof): style {'n/a' if h is None else f'{h}%'}, "
            f"model {'n/a' if likelihood is None else f'{likelihood}%'}")
    reasons = [scrub(x) for x in llm_view.get("reasons") or [] if str(x).strip()]
    if reasons:
        text += f' - "{_clip(reasons[0])}"'
    return _signal(level, "ai", text)


def _code_signal(report: dict, uid: str, scrub) -> dict | None:
    r = ((report.get("code") or {}).get("results") or {}).get(uid)
    if not r:
        return None
    review = r.get("llm") or {}
    passed, total = r.get("passed") or 0, r.get("total") or 0
    if r.get("error"):
        return _signal(MEDIUM, "code", f"code could not be checked: {_clip(scrub(r['error']))}")
    if total and passed / total < 0.5:
        level = MEDIUM
    elif review and review.get("solves_task") is False:
        level = MEDIUM
    elif total and passed < total:
        level = INFO
    else:
        return None
    text = f"code tests {passed}/{total} passed"
    if review.get("summary"):
        text += f"; reviewer: {_clip(scrub(review['summary']))}"
    return _signal(level, "code", text)


def _truncated(report: dict, uid: str) -> list[str]:
    found = set()
    for section in ("numbers", "rubric"):
        r = ((report.get(section) or {}).get("results") or {}).get(uid) or {}
        found.update(r.get("truncated") or [])
    return sorted(found)


def student_signals(report: dict, uid: str, code_of: dict, scrub) -> list[dict]:
    s = report["students"][uid]
    out: list[dict] = []
    if not s.get("files"):
        out.append(_signal(HIGH, "unreadable",
                           "nothing could be read, so the work is unmarked; ask for a readable "
                           "format (Google Doc, .docx, .py)"))
    out += _copy_signals(report, uid, code_of, scrub)
    sc = (report.get("scores") or {}).get(uid) or results.score(report, uid)
    if sc and sc["percent"] < LOW_SCORE:
        lost = [q for q, v in sorted(sc["per_question"].items()) if v < 0.999]
        out.append(_signal(MEDIUM, "score", f"low score {sc['percent']}%"
                           + (f"; lost marks on Q{', Q'.join(lost)}" if lost else "")))
    trimmed = _truncated(report, uid)
    if trimmed:
        out.append(_signal(MEDIUM, "truncated",
                           f"part of the work was cut to fit the model ({', '.join(trimmed)}); "
                           "check those marks by hand"))
    for sig in (_ai_signal(report, uid, scrub), _code_signal(report, uid, scrub)):
        if sig:
            out.append(sig)
    if s.get("status") == st.LATE:
        out.append(_signal(INFO, "late", "turned in late"))
    if s.get("files") and s.get("notes"):
        out.append(_signal(INFO, "files", f"{len(s['notes'])} attachment(s) not checked: "
                                          + _clip(scrub("; ".join(s["notes"])))))
    order = {HIGH: 0, MEDIUM: 1, INFO: 2}
    out.sort(key=lambda x: order[x["level"]])
    return out


def tier_of(signals: list[dict]) -> str:
    levels = {s["level"] for s in signals}
    return ACT if HIGH in levels else REVIEW if MEDIUM in levels else OK


# --- The brief ---------------------------------------------------------------

def _class_stats(report: dict) -> dict:
    scores = [sc for sc in (report.get("scores") or {}).values() if sc]
    percents = [sc["percent"] for sc in scores]
    per_q: dict[str, list[float]] = {}
    for sc in scores:
        for q, v in sc["per_question"].items():
            per_q.setdefault(q, []).append(v)
    missed = sorted(((q, mean(v)) for q, v in per_q.items()), key=lambda kv: kv[1])[:3]
    code = ((report.get("code") or {}).get("results") or {}).values()
    return {
        "students": len(report.get("students") or {}),
        "scored": len(scores),
        "mean": round(mean(percents), 1) if percents else None,
        "median": round(median(percents), 1) if percents else None,
        "bands": {"under_50": sum(p < 50 for p in percents),
                  "50_79": sum(50 <= p < 80 for p in percents),
                  "80_up": sum(p >= 80 for p in percents)},
        "most_missed": [{"question": q, "right": round(100 * v)} for q, v in missed if v < 0.999],
        "code_all_passed": sum(1 for r in code if r.get("total") and r.get("passed") == r.get("total")),
        "code_checked": sum(1 for r in code if r.get("total")),
    }


def brief(report: dict, report_file: str = "") -> dict:
    """Everything the agent needs to triage a run, with students as codes only."""
    code_of = codes(report)
    scrub = _scrubber(report, code_of)
    work = report.get("work") or {}
    people = []
    for uid, s in report.get("students", {}).items():
        sigs = student_signals(report, uid, code_of, scrub)
        sc = (report.get("scores") or {}).get(uid) or results.score(report, uid)
        people.append({"code": code_of[uid], "tier": tier_of(sigs),
                       "score": sc["percent"] if sc else None,
                       "status": st.LABELS.get(s.get("status"), s.get("status") or ""),
                       "signals": sigs})
    rank = {ACT: 0, REVIEW: 1, OK: 2}
    people.sort(key=lambda p: (rank[p["tier"]], -len(p["signals"]),
                               p["score"] if p["score"] is not None else -1, p["code"]))
    stats = _class_stats(report)
    stats["copy_pairs"] = sum(1 for p in people for s in p["signals"]
                              if s["kind"] == "copying" and s["level"] != INFO) // 2
    stats["ai_flagged"] = sum(1 for p in people for s in p["signals"]
                              if s["kind"] == "ai" and s["level"] == MEDIUM)
    return {
        "report": report_file,
        "assignment": work.get("title", ""),
        "course": (report.get("course") or {}).get("name", ""),
        "checked_at": report.get("generated_at", ""),
        "checks": (report.get("options") or {}).get("checks") or [],
        "grading": {"max_points": work.get("max_points"),
                    "can_save_draft_grades": work.get("associated_with_developer")},
        "stats": stats,
        "students": people,
        "errors": [_clip(scrub(e), 240) for e in report.get("errors") or []],
    }


def _score_text(score: float | None) -> str:
    return "unmarked" if score is None else f"{score}%"


def brief_markdown(b: dict) -> str:
    """The brief as compact text for a small local model (and for humans)."""
    s = b["stats"]
    g = b["grading"]
    lines = [f"# Triage brief: {b['assignment']}",
             f"Course: {b['course'] or '-'} · checked {b['checked_at'][:16].replace('T', ' ')} · "
             f"{s['students']} students · checks: {', '.join(b['checks']) or '-'}"]
    if g["max_points"]:
        saveable = {True: "yes", False: "no (assignment was created outside this app)",
                    None: "unknown"}[g["can_save_draft_grades"]]
        lines.append(f"Grading: out of {g['max_points']} points; draft grades can be saved to Classroom: {saveable}")
    lines += ["", "## Class summary"]
    if s["scored"]:
        lines.append(f"- Scored {s['scored']} of {s['students']}: mean {s['mean']}%, median {s['median']}%; "
                     f"under 50%: {s['bands']['under_50']}, 50-79%: {s['bands']['50_79']}, "
                     f"80%+: {s['bands']['80_up']}")
        if s["most_missed"]:
            lines.append("- Most missed: " + ", ".join(f"Q{m['question']} ({m['right']}% right)"
                                                       for m in s["most_missed"]))
    else:
        lines.append("- No scores (no answer sheet, or nothing could be marked)")
    lines.append(f"- Copying pairs flagged: {s['copy_pairs']} · AI-writing flags: {s['ai_flagged']}"
                 + (f" · code: {s['code_all_passed']} of {s['code_checked']} passed every test"
                    if s["code_checked"] else ""))
    for tier in (ACT, REVIEW):
        group = [p for p in b["students"] if p["tier"] == tier]
        if not group:
            continue
        lines += ["", f"## {TIER_LABELS[tier]} ({len(group)})"]
        for p in group:
            lines.append(f"- {p['code']} · {_score_text(p['score'])} · {p['status']}")
            lines += [f"  - [{sig['level']}] {sig['kind']}: {sig['text']}" for sig in p["signals"]]
    fine = [p for p in b["students"] if p["tier"] == OK]
    if fine:
        lines += ["", f"## {TIER_LABELS[OK]} ({len(fine)})",
                  ", ".join(f"{p['code']} ({_score_text(p['score'])})" for p in fine)]
    if b["errors"]:
        lines += ["", "## Problems during the run"] + [f"- {e}" for e in b["errors"]]
    return "\n".join(lines)


def student_detail(report: dict, code: str) -> dict | None:
    """One student's full picture, still as a code: marks per question, tests, signals."""
    code_of = codes(report)
    uid = next((u for u, c in code_of.items() if c.lower() == code.strip().lower()), None)
    if uid is None:
        return None
    scrub = _scrubber(report, code_of)
    s = report["students"][uid]
    sc = (report.get("scores") or {}).get(uid) or results.score(report, uid)
    nums = ((report.get("numbers") or {}).get("results") or {}).get(uid) or {}
    rub = ((report.get("rubric") or {}).get("results") or {}).get(uid) or {}
    code_r = ((report.get("code") or {}).get("results") or {}).get(uid) or {}
    ai = (report.get("ai") or {}).get(uid) or {}
    sigs = student_signals(report, uid, code_of, scrub)
    return {
        "code": code_of[uid], "tier": tier_of(sigs),
        "status": st.LABELS.get(s.get("status"), s.get("status") or ""),
        "files": len(s.get("files") or []),
        "score": sc,
        "numeric": [{"question": r["question"], "ok": r["ok"]} for r in nums.get("results") or []],
        "written": [{"question": r["question"], "covered": len(r.get("covered") or []),
                     "total": r.get("total"), "missing": [scrub(m) for m in r.get("missing") or []]}
                    for r in rub.get("results") or []],
        "code_tests": {"passed": code_r.get("passed"), "total": code_r.get("total"),
                       "issues": [scrub(i) for i in (code_r.get("llm") or {}).get("issues") or []]}
        if code_r else None,
        "ai_style_signals": [scrub(x) for x in (ai.get("heuristic") or {}).get("signals") or []],
        "signals": sigs,
    }


# --- Agent output stored next to the report ----------------------------------

def _agent_dir(report_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(report_path)), "agent")


def triage_path(report_path: str) -> str:
    stem = os.path.basename(report_path)[:-len(".json")]
    return os.path.join(_agent_dir(report_path), stem + ".triage.json")


def load_triage(report_path: str) -> dict | None:
    try:
        with open(triage_path(report_path), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_triage(report_path: str, **fields) -> dict:
    """Merge fields into the stored triage state (state, text, hold, error ...)."""
    data = load_triage(report_path) or {}
    data.update(fields, updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    os.makedirs(_agent_dir(report_path), exist_ok=True)
    tmp = triage_path(report_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, triage_path(report_path))      # never leave a half-written file
    return data


def parse_codes(text: str, report: dict) -> list[str]:
    """The valid student codes mentioned in `text` ("S03, S07", '["S03"]', ...)."""
    valid = set(codes(report).values())
    seen = []
    for code in CODE_RE.findall(text or ""):
        if code in valid and code not in seen:
            seen.append(code)
    return seen


def with_names(text: str, report: dict) -> str:
    """Put names back for the teacher: 'S03' -> 'Nguyen Van An (S03)'."""
    names = {c: report["students"][uid]["name"] for uid, c in codes(report).items()}
    return CODE_RE.sub(lambda m: f"{names[m.group(0)]} ({m.group(0)})" if m.group(0) in names
                       else m.group(0), text or "")
