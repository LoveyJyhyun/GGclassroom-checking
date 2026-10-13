"""Final score from the answer sheet, and the CSV export of a check report.

The score comes **only** from the answer sheet: numeric questions are right or
wrong, written questions get partial credit for the points they cover. Copying,
AI indicators and code test results are reported next to it but never folded into
the mark - they are signals for a teacher to act on, not marks.
"""
from __future__ import annotations

import csv
import io


def question_scores(report: dict, uid: str) -> dict[str, float]:
    """Question label -> fraction earned (0.0 - 1.0) for one student."""
    parts: dict[str, list[float]] = {}

    nums = ((report.get("numbers") or {}).get("results") or {}).get(uid)
    if nums:
        for r in nums["results"]:
            parts.setdefault(str(r["question"]), []).append(1.0 if r["ok"] else 0.0)

    rub = ((report.get("rubric") or {}).get("results") or {}).get(uid)
    if rub:
        for r in rub["results"]:
            if r["total"]:
                parts.setdefault(str(r["question"]), []).append(len(r["covered"]) / r["total"])

    # A label checked both ways (a number and required points) counts once.
    return {label: sum(v) / len(v) for label, v in parts.items()}


def score(report: dict, uid: str) -> dict | None:
    """Final mark for one student, or None when there is nothing to mark against.

    A student whose attachments could not be read at all (a .rar, a photo of their
    work) gets **no** mark rather than zero: nothing was seen, so nothing is known.
    Scoring them 0 would put a mark in the CSV that the teacher might enter as real.
    """
    student = report.get("students", {}).get(uid) or {}
    if not student.get("files"):
        return None

    per = question_scores(report, uid)
    if not per:
        return None
    earned = sum(per.values())
    return {"percent": round(100 * earned / len(per), 1), "earned": round(earned, 2),
            "questions": len(per), "per_question": per}


def scores(report: dict) -> dict[str, dict | None]:
    return {uid: score(report, uid) for uid in report.get("students", {})}


def question_labels(report: dict) -> list[str]:
    """Every question label the report marks, numeric ones first, in key order."""
    labels = list((report.get("numbers") or {}).get("key", {}))
    for q in (report.get("rubric") or {}).get("questions", []):
        if q["label"] not in labels:
            labels.append(q["label"])
    return labels


def _copy_partner(report: dict, uid: str, names: dict[str, str]) -> tuple[str, str]:
    """The most similar other student, if any pair was flagged."""
    best, who = -1.0, ""
    for p in report.get("duplicates") or []:
        if uid not in (p["a"], p["b"]):
            continue
        strength = max(p["lexical"] or 0, p["code"] or 0, p["semantic"] or 0)
        if strength > best:
            other = p["b"] if p["a"] == uid else p["a"]
            best, who = strength, names.get(other, other)
    return (f"{best:.0%}" if best >= 0 else ""), who


HEADERS = ["Student", "Email", "Status", "Score %", "Questions", "Read",
           "Answers correct", "Answers total", "Written points", "Written total",
           "AI heuristic %", "AI model %", "Code tests passed", "Code tests total",
           "Copy similarity", "Copy with", "Truncated", "Notes"]


def to_csv(report: dict) -> str:
    """One row per student: the mark, then every signal, then per-question detail."""
    students = report.get("students", {})
    names = {uid: s["name"] for uid, s in students.items()}
    labels = question_labels(report)

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(HEADERS + [f"Q{label}" for label in labels])

    for uid, student in sorted(students.items(), key=lambda kv: kv[1]["name"].lower()):
        sc = score(report, uid)
        readable = bool(student.get("files"))
        # Nothing was read, so "0 of 2 correct" would be just as false as a 0% score.
        nums = ((report.get("numbers") or {}).get("results") or {}).get(uid) or {} if readable else {}
        rub = ((report.get("rubric") or {}).get("results") or {}).get(uid) or {} if readable else {}
        ai = (report.get("ai") or {}).get(uid) or {} if readable else {}
        code = ((report.get("code") or {}).get("results") or {}).get(uid) or {} if readable else {}
        similarity, partner = _copy_partner(report, uid, names) if readable else ("", "")

        trimmed = sorted({s for src in (nums, rub, code.get("llm") or {}, ai.get("llm") or {})
                          for s in (src.get("truncated") or [])})

        row = [
            student["name"], student.get("email", ""), student.get("status", ""),
            sc["percent"] if sc else "", sc["questions"] if sc else "",
            # Blank score + "nothing readable" is the difference between "scored 0"
            # and "we could not open the file" - never let those look the same.
            "yes" if readable else "NOTHING READABLE",
            nums.get("correct", ""), nums.get("total", ""),
            rub.get("covered", ""), rub.get("total", ""),
            (ai.get("heuristic") or {}).get("score", ""),
            (ai.get("llm") or {}).get("likelihood", ""),
            code.get("passed", ""), code.get("total", ""),
            similarity, partner,
            "; ".join(trimmed),
            "; ".join(student.get("notes") or []),
        ]
        per = sc["per_question"] if sc else {}
        for label in labels:
            value = per.get(label)
            row.append("" if value is None else round(value, 2))
        writer.writerow(row)

    return out.getvalue()


def save_csv(report: dict, path: str) -> str:
    # utf-8-sig so Excel on Windows shows Vietnamese names correctly.
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(to_csv(report))
    return path


def demo() -> None:
    """Self-check: a numeric question is all-or-nothing, a written one gets part marks."""
    report = {
        "students": {
            "a": {"name": "An", "email": "a@s", "status": "Turned in",
                  "notes": [], "files": ["bai.docx"]},
            "z": {"name": "Zung", "email": "z@s", "status": "Turned in",
                  "notes": ["bai.rar: RAR archives need WinRAR."], "files": []},
        },
        "numbers": {"key": {"1": {"value": 42}}, "results": {
            "a": {"results": [{"question": "1", "ok": True, "got": 42}], "correct": 1, "total": 1}}},
        "rubric": {"questions": [{"label": "3", "points": ["x", "y", "z", "w"]}], "results": {
            "a": {"results": [{"question": "3", "covered": ["x"], "missing": ["y", "z", "w"],
                               "total": 4, "score": 25}], "covered": 1, "total": 4}}},
        "duplicates": [], "ai": None, "code": None,
    }
    sc = score(report, "a")
    assert sc["per_question"] == {"1": 1.0, "3": 0.25}, sc
    assert sc["percent"] == 62.5, sc            # (1.0 + 0.25) / 2 questions
    rows = to_csv(report).splitlines()
    assert rows[0].endswith("Q1,Q3"), rows[0]
    assert rows[1].startswith("An,a@s,Turned in,62.5,2,yes,1,1,1,4"), rows[1]

    # A student whose file could not be opened must not be marked zero.
    assert score(report, "z") is None
    unread = [r for r in rows if r.startswith("Zung")][0]
    assert unread.startswith("Zung,z@s,Turned in,,,NOTHING READABLE"), unread
    print(to_csv(report))
    print("OK: marks follow the answer sheet; an unreadable file scores nothing, not zero")


if __name__ == "__main__":
    demo()
