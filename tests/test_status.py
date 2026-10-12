from datetime import datetime, timezone

from checker import status as st

DUE = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
BEFORE = datetime(2026, 2, 28, tzinfo=timezone.utc)
AFTER = datetime(2026, 3, 2, tzinfo=timezone.utc)


def test_due_datetime():
    work = {"dueDate": {"year": 2026, "month": 3, "day": 1}, "dueTime": {"hours": 12}}
    assert st.due_datetime(work) == DUE
    assert st.due_datetime({}) is None


def test_statuses():
    assert st.submission_status({"state": "TURNED_IN"}, DUE, BEFORE) == st.TURNED_IN
    assert st.submission_status({"state": "TURNED_IN", "late": True}, DUE, AFTER) == st.LATE
    assert st.submission_status({"state": "RETURNED", "assignedGrade": 9}, DUE, AFTER) == st.GRADED
    assert st.submission_status({"state": "RETURNED"}, DUE, AFTER) == st.RETURNED
    assert st.submission_status({"state": "CREATED"}, DUE, BEFORE) == st.ASSIGNED
    assert st.submission_status({"state": "CREATED"}, DUE, AFTER) == st.MISSING
    assert st.submission_status({"state": "RECLAIMED_BY_STUDENT"}, None, AFTER) == st.ASSIGNED


def test_turned_in_at_uses_latest():
    sub = {"submissionHistory": [
        {"stateHistory": {"state": "CREATED", "stateTimestamp": "a"}},
        {"stateHistory": {"state": "TURNED_IN", "stateTimestamp": "b"}},
        {"gradeHistory": {}},
        {"stateHistory": {"state": "TURNED_IN", "stateTimestamp": "c"}},
    ]}
    assert st.turned_in_at(sub) == "c"


def test_attachments():
    sub = {"assignmentSubmission": {"attachments": [
        {"driveFile": {"id": "f1", "title": "Essay", "alternateLink": "u1"}},
        {"link": {"url": "http://x"}},
    ]}}
    atts = st.attachments_of(sub)
    assert atts[0]["file_id"] == "f1" and atts[1]["kind"] == "link"
