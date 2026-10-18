"""Draft grades proposed from the answer-sheet score, saved only after the teacher approves.

A proposal is plain arithmetic: score % x the assignment's points, rounded to
GRADE_STEP. The agent can ask for a proposal and say whose grade to hold back (for
example a likely copy), but it cannot approve or save one. Saving happens only when
the teacher presses the button on the report page, in their own signed-in session.

Once any grade has been saved, the proposal is locked: neither the agent nor a new
proposal from the page can replace it, so the record of what was written to
Classroom is never lost. Every change gets a new revision, and the page sends the
revision it showed, so the teacher can only approve the exact numbers they saw.

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
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from googleapiclient.errors import HttpError

from . import results
from . import status as st

GRADE_STEP = float(os.environ.get("GRADE_STEP", "0.1"))
PENDING, APPLIED, PARTIAL = "pending", "applied", "partial"
AGENT_HOLD = "held by the agent: "


class ProposalLocked(ValueError):
    """Grades from this proposal were already saved to Classroom."""


class StaleProposal(ValueError):
    """The proposal changed after the page that is approving it was opened."""


# Proposal files are read, changed and written back by the web thread, the triage
# thread and MCP calls; one lock per report keeps those from overwriting each other.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def lock_for(report_path: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(os.path.abspath(report_path), threading.Lock())


def round_to_step(value: float, step: float = GRADE_STEP) -> float:
    """Round half up to the nearest step, exactly (4.35 -> 4.4, not 4.3 by float error)."""
    if step <= 0:
        return value
    d_step = Decimal(str(step))
    units = (Decimal(str(value)) / d_step).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return float(units * d_step)


def scores_of(report: dict) -> dict:
    """Scores stored in the report, or worked out for reports saved without them."""
    return report.get("scores") or results.scores(report)


def is_locked(proposal: dict | None) -> bool:
    return bool(proposal) and any(r.get("result") == "saved" for r in proposal["rows"])


def propose(report: dict, max_points: float | None = None, hold: dict[str, str] | None = None,
            created_by: str = "teacher") -> dict:
    """Proposed draft grade per marked student. `hold` maps uid -> reason to hold back."""
    work = report.get("work") or {}
    max_points = max_points if max_points is not None else work.get("max_points")
    if not max_points or not math.isfinite(max_points) or max_points <= 0 or max_points > 100_000:
        raise ValueError("This assignment has no usable point value, so there is no grade to "
                         "propose. Enter the maximum points.")
    hold = hold or {}
    scores = scores_of(report)
    rows, unmarked = [], []
    students = report.get("students") or {}
    for uid, s in sorted(students.items(), key=lambda kv: kv[1]["name"].lower()):
        sc = scores.get(uid)
        if not sc:
            if s.get("status") not in st.DONE:
                why = "not turned in"
            else:
                why = "nothing readable" if not s.get("files") else "nothing to mark against"
            unmarked.append({"uid": uid, "name": s["name"], "why": why})
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
        "revision": uuid.uuid4().hex[:12],
        "course_id": report.get("course_id"), "work_id": work.get("id"),
        "work_title": work.get("title", ""),
        "max_points": max_points, "step": GRADE_STEP,
        "can_save": work.get("associated_with_developer"),
        "rows": rows, "unmarked": unmarked,
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def merge_agent_holds(proposal: dict, hold: dict[str, str]) -> dict:
    """Update which rows the agent holds, keeping the teacher's points and everything else."""
    for row in proposal["rows"]:
        if row["uid"] in hold:
            row["held"], row["reason"] = True, hold[row["uid"]]
        elif row["held"] and row["reason"].startswith(AGENT_HOLD):
            row["held"], row["reason"] = False, ""        # the agent changed its mind
    proposal["revision"] = uuid.uuid4().hex[:12]
    proposal["agent_updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return proposal


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
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(proposal, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)                         # never leave a half-written file
    return path


def replace(report_path: str, proposal: dict) -> None:
    """Store a new proposal unless grades from the current one were already saved."""
    with lock_for(report_path):
        if is_locked(load(report_path)):
            raise ProposalLocked("Draft grades were already saved for this report.")
        save(report_path, proposal)


# --- Applying an approved proposal -------------------------------------------

def _error_text(e: Exception) -> str:
    if isinstance(e, HttpError):
        if e.status_code == 403:
            return ("Google refused (403). Grades can only be written on assignments this app "
                    "created, by a teacher signed in with grade access.")
        return f"Google error {e.status_code}"
    return f"{type(e).__name__}: {e}"


def _status(proposal: dict) -> str:
    saved = any(r.get("result") == "saved" for r in proposal["rows"])
    failed = any((r.get("result") or "").startswith("error") for r in proposal["rows"])
    return (PARTIAL if failed else APPLIED) if saved else PENDING


def apply(cr, proposal: dict, approved: set[str]) -> dict:
    """Write the approved rows to Classroom as draft grades. Returns the updated proposal.

    Each row is attempted on its own and records its own result, so one refusal
    does not hide the others, a retry only resends what failed, and a row that was
    saved stays saved whatever is ticked next time.
    """
    tried = failed = 0
    for row in proposal["rows"]:
        if row.get("result") == "saved":
            continue
        if row["uid"] not in approved:
            row["result"] = "held" if row["held"] else "skipped"
            continue
        tried += 1
        try:
            cr.set_draft_grade(proposal["course_id"], proposal["work_id"], row["submission_id"], row["grade"])
            row["result"] = "saved"
        except Exception as e:                     # noqa: BLE001 - recorded per row
            row["result"] = "error: " + _error_text(e)
            failed += 1
    proposal["status"] = _status(proposal)
    proposal["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proposal["applied_counts"] = {
        "saved": sum(r.get("result") == "saved" for r in proposal["rows"]),
        "tried": tried, "failed": failed}
    proposal["revision"] = uuid.uuid4().hex[:12]
    return proposal


def apply_reviewed(cr, report_path: str, revision: str, approved: set[str]) -> dict:
    """apply() under the report's lock, only if the teacher saw this exact revision."""
    with lock_for(report_path):
        proposal = load(report_path)
        if proposal is None:
            raise StaleProposal("There is no draft-grade proposal for this report.")
        if (proposal.get("revision") or "") != (revision or ""):
            raise StaleProposal("The proposal changed after you opened the page (the triage agent "
                                "may have updated it). Check it again, then save.")
        proposal = apply(cr, proposal, approved)
        save(report_path, proposal)
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
