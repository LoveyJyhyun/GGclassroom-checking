"""Draft grades proposed from the answer-sheet score, saved only after the teacher approves.

A proposal is plain arithmetic: score % x the assignment's points, rounded to
GRADE_STEP. The agent can ask for a proposal and say whose grade to hold back (for
example a likely copy), but it cannot approve or save one. Saving happens only when
the teacher presses the button on the report page, in their own signed-in session.

Google only lets an app write grades on assignments created by the same Google
Cloud project ("associatedWithDeveloper"). For assignments made in the normal
Classroom page the proposal is still useful: download it as CSV and type the
grades in.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from datetime import datetime, timezone

from googleapiclient.errors import HttpError

GRADE_STEP = float(os.environ.get("GRADE_STEP", "0.1"))
PENDING, APPLIED, PARTIAL = "pending", "applied", "partial"


def round_to_step(value: float, step: float = GRADE_STEP) -> float:
    """Round half up to the nearest step, without float noise (7.300000000000001)."""
    if step <= 0:
        return value
    rounded = math.floor(value / step + 0.5) * step
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    return round(rounded, decimals + 2)


def propose(report: dict, max_points: float | None = None, hold: dict[str, str] | None = None,
            created_by: str = "teacher") -> dict:
    """Proposed draft grade per marked student. `hold` maps uid -> reason to hold back."""
    work = report.get("work") or {}
    max_points = max_points if max_points is not None else work.get("max_points")
    if not max_points or max_points <= 0:
        raise ValueError("This assignment has no point value, so there is no grade to propose. "
                         "Enter the maximum points.")
    hold = hold or {}
    rows, unmarked = [], []
    students = report.get("students") or {}
    for uid, s in sorted(students.items(), key=lambda kv: kv[1]["name"].lower()):
        sc = (report.get("scores") or {}).get(uid)
        if not sc:
            unmarked.append({"uid": uid, "name": s["name"],
                             "why": "nothing readable" if not s.get("files") else "nothing to mark against"})
            continue
        rows.append({
            "uid": uid, "name": s["name"], "email": s.get("email", ""),
            "submission_id": s.get("submission_id", ""),
            "percent": sc["percent"],
            "grade": round_to_step(sc["percent"] / 100 * max_points),
            "held": uid in hold, "reason": hold.get(uid, ""),
            "result": None,
        })
    return {
        "status": PENDING,
        "course_id": report.get("course_id"), "work_id": work.get("id"),
        "work_title": work.get("title", ""),
        "max_points": max_points, "step": GRADE_STEP,
        "can_save": work.get("associated_with_developer"),
        "rows": rows, "unmarked": unmarked,
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --- Storage, next to the report ---------------------------------------------

def proposal_path(report_path: str) -> str:
    stem = os.path.basename(report_path)[:-len(".json")]
    return os.path.join(os.path.dirname(os.path.abspath(report_path)), "agent", stem + ".grades.json")


def load(report_path: str) -> dict | None:
    try:
        with open(proposal_path(report_path), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save(report_path: str, proposal: dict) -> str:
    path = proposal_path(report_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(proposal, fh, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    return path


# --- Applying an approved proposal -------------------------------------------

def _error_text(e: Exception) -> str:
    if isinstance(e, HttpError):
        if e.status_code == 403:
            return ("Google refused (403). Grades can only be written on assignments this app "
                    "created, by a teacher signed in with grade access.")
        return f"Google error {e.status_code}"
    return f"{type(e).__name__}: {e}"


def apply(cr, proposal: dict, approved: set[str]) -> dict:
    """Write the approved rows to Classroom as draft grades. Returns the updated proposal.

    Each row is attempted on its own and records its own result, so one refusal
    does not hide the others and a retry only resends what failed.
    """
    saved = failed = 0
    for row in proposal["rows"]:
        if row["uid"] not in approved:
            if row.get("result") != "saved":
                row["result"] = "held" if row["held"] else "skipped"
            continue
        if row.get("result") == "saved":
            saved += 1
            continue
        try:
            cr.set_draft_grade(proposal["course_id"], proposal["work_id"], row["submission_id"], row["grade"])
            row["result"] = "saved"
            saved += 1
        except Exception as e:                     # noqa: BLE001 - recorded per row
            row["result"] = "error: " + _error_text(e)
            failed += 1
    proposal["status"] = APPLIED if saved and not failed else PARTIAL if saved else PENDING
    proposal["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proposal["applied_counts"] = {"saved": saved, "failed": failed}
    return proposal


def to_csv(proposal: dict) -> str:
    """For typing grades into Classroom by hand when the app may not write them."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["Student", "Email", "Score %", "Proposed grade", "Out of", "Hold", "Reason", "Saved"])
    for row in proposal["rows"]:
        writer.writerow([row["name"], row["email"], row["percent"], row["grade"],
                         proposal["max_points"], "yes" if row["held"] else "", row["reason"],
                         row.get("result") or ""])
    for row in proposal["unmarked"]:
        writer.writerow([row["name"], "", "", "", proposal["max_points"], "", row["why"], ""])
    return out.getvalue()
