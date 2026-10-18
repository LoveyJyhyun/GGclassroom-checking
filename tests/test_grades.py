import pytest
from googleapiclient.errors import HttpError

from checker import grades


class Resp(dict):
    status = 403
    reason = "Forbidden"


class FakeClassroom:
    def __init__(self, refuse=()):
        self.saved, self.refuse = [], set(refuse)

    def set_draft_grade(self, course_id, work_id, sub_id, grade):
        if sub_id in self.refuse:
            raise HttpError(Resp(), b'{"error": {"message": "ProjectPermissionDenied"}}')
        self.saved.append((course_id, work_id, sub_id, grade))
        return {}


def test_rounding():
    assert grades.round_to_step(7.25, 0.1) == 7.3
    assert grades.round_to_step(4.35, 0.1) == 4.4 and grades.round_to_step(0.35, 0.1) == 0.4
    assert grades.round_to_step(6.25, 0.1) == 6.3
    assert grades.round_to_step(7.249, 0.5) == 7.0
    assert grades.round_to_step(8.3333, 0.25) == 8.25
    assert grades.round_to_step(6.5, 1) == 7


def test_proposal_scales_the_score_and_skips_unmarked(sample_report):
    p = grades.propose(sample_report, hold={"b": "likely copied"})
    rows = {r["uid"]: r for r in p["rows"]}
    assert set(rows) == {"a", "b", "d"}
    assert rows["a"]["grade"] == 10 and rows["b"]["grade"] == 5
    assert rows["b"]["held"] and rows["b"]["reason"] == "likely copied"
    assert p["unmarked"] == [{"uid": "c", "name": "Lê Chi", "why": "nothing readable"}]
    assert p["status"] == grades.PENDING and p["can_save"] is True


def test_no_points_no_proposal(sample_report):
    sample_report["work"]["max_points"] = None
    with pytest.raises(ValueError):
        grades.propose(sample_report)
    for bad in (float("inf"), float("nan"), -5):
        with pytest.raises(ValueError):
            grades.propose(sample_report, max_points=bad)
    assert grades.propose(sample_report, max_points=20)["rows"][0]["grade"] == 20


def test_only_approved_rows_are_saved(sample_report):
    p = grades.propose(sample_report, hold={"b": "check first"})
    cr = FakeClassroom()
    grades.apply(cr, p, approved={"a", "d"})
    assert cr.saved == [("c1", "w1", "sa", 10.0), ("c1", "w1", "sd", 10.0)]
    assert {r["uid"]: r["result"] for r in p["rows"]} == {"a": "saved", "b": "held", "d": "saved"}
    assert p["status"] == grades.APPLIED

    # Approving the held one later sends only that one.
    grades.apply(cr, p, approved={"a", "b", "d"})
    assert cr.saved[-1] == ("c1", "w1", "sb", 5.0) and len(cr.saved) == 3


def test_a_refusal_is_recorded_per_student(sample_report):
    p = grades.propose(sample_report)
    grades.apply(FakeClassroom(refuse={"sd"}), p, approved={"a", "b", "d"})
    results = {r["uid"]: r["result"] for r in p["rows"]}
    assert results["a"] == results["b"] == "saved"
    assert results["d"].startswith("error: Google refused (403)")
    assert p["status"] == grades.PARTIAL


def test_storage_and_csv(tmp_path, sample_report):
    path = str(tmp_path / "r.json")
    assert grades.load(path) is None
    grades.save(path, grades.propose(sample_report, hold={"b": "x"}))
    text = grades.to_csv(grades.load(path))
    lines = text.splitlines()
    assert lines[0].startswith("Student,Email,Score %,Proposed grade,Out of,Hold")
    assert any(line.startswith("Trần Bình,binh@school.test,50.0,5.0,10,yes,x") for line in lines)
    assert any(line.startswith("Lê Chi,") and "nothing readable" in line for line in lines)


def test_saved_grades_lock_the_proposal(tmp_path, sample_report):
    path = str(tmp_path / "r.json")
    p = grades.propose(sample_report)
    grades.save(path, p)
    grades.apply_reviewed(FakeClassroom(), path, p["revision"], {"a"})
    locked = grades.load(path)
    assert locked["status"] == grades.APPLIED and grades.is_locked(locked)
    # Pressing Save again with nothing ticked keeps the record of what was saved.
    locked = grades.apply_reviewed(FakeClassroom(), path, locked["revision"], set())
    assert locked["status"] == grades.APPLIED
    with pytest.raises(grades.ProposalLocked):
        grades.replace(path, grades.propose(sample_report))
    with pytest.raises(grades.StaleProposal):
        grades.apply_reviewed(FakeClassroom(), path, "not-the-revision", {"b"})
