"""Pure logic for classifying Google Classroom submissions.

Kept free of API calls so it can be unit-tested.
"""
from __future__ import annotations

from datetime import datetime, timezone

GRADED = "graded"
RETURNED = "returned"
TURNED_IN = "turned_in"
LATE = "late"
MISSING = "missing"
ASSIGNED = "assigned"

LABELS = {
    GRADED: "Graded",
    RETURNED: "Returned",
    TURNED_IN: "Turned in",
    LATE: "Turned in late",
    MISSING: "Missing",
    ASSIGNED: "Not turned in yet",
}

# Statuses that count as "the student handed something in".
DONE = {GRADED, RETURNED, TURNED_IN, LATE}


def due_datetime(course_work: dict) -> datetime | None:
    """Return the coursework due time as an aware UTC datetime, or None."""
    date = course_work.get("dueDate")
    if not date:
        return None
    # The API omits zero-valued fields, so {"hours": 12} means 12:00.
    time = course_work.get("dueTime")
    hours, minutes = (time.get("hours", 0), time.get("minutes", 0)) if time is not None else (23, 59)
    return datetime(date["year"], date["month"], date["day"], hours, minutes, tzinfo=timezone.utc)


def submission_status(submission: dict, due: datetime | None, now: datetime | None = None) -> str:
    """Classify one studentSubmission resource."""
    now = now or datetime.now(timezone.utc)
    state = submission.get("state")
    if state == "RETURNED":
        return GRADED if submission.get("assignedGrade") is not None else RETURNED
    if state == "TURNED_IN":
        return LATE if submission.get("late") else TURNED_IN
    # NEW, CREATED, RECLAIMED_BY_STUDENT: nothing currently handed in.
    if submission.get("late") or (due is not None and now > due):
        return MISSING
    return ASSIGNED


def turned_in_at(submission: dict) -> str | None:
    """Timestamp of the most recent TURNED_IN transition, if any."""
    latest = None
    for entry in submission.get("submissionHistory", []):
        state = entry.get("stateHistory")
        if state and state.get("state") == "TURNED_IN":
            latest = state.get("stateTimestamp")
    return latest


def grade_of(submission: dict) -> float | None:
    grade = submission.get("assignedGrade")
    return grade if grade is not None else submission.get("draftGrade")


def answer_text(submission: dict) -> str | None:
    """The student's answer for question-type coursework."""
    if "shortAnswerSubmission" in submission:
        return submission["shortAnswerSubmission"].get("answer")
    if "multipleChoiceSubmission" in submission:
        return submission["multipleChoiceSubmission"].get("answer")
    return None


def attachments_of(submission: dict) -> list[dict]:
    """Normalise attachments into {kind, title, url, file_id} dicts."""
    result = []
    for att in submission.get("assignmentSubmission", {}).get("attachments", []):
        if "driveFile" in att:
            f = att["driveFile"]
            result.append({"kind": "drive", "title": f.get("title", "Drive file"),
                           "url": f.get("alternateLink"), "file_id": f.get("id")})
        elif "link" in att:
            link = att["link"]
            result.append({"kind": "link", "title": link.get("title") or link.get("url"),
                           "url": link.get("url"), "file_id": None})
        elif "youTubeVideo" in att:
            v = att["youTubeVideo"]
            result.append({"kind": "youtube", "title": v.get("title", "YouTube video"),
                           "url": v.get("alternateLink"), "file_id": None})
        elif "form" in att:
            form = att["form"]
            result.append({"kind": "form", "title": form.get("title", "Form"),
                           "url": form.get("formUrl"), "file_id": None})
    return result
