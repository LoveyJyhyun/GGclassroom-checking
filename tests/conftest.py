import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import copy

import pytest

from checker import results as _results

_SAMPLE = {
    "generated_at": "2026-10-13T13:00:00+00:00", "course_id": "c1",
    "course": {"id": "c1", "name": "Science"},
    "work": {"id": "w1", "title": "Lab 1", "max_points": 10, "associated_with_developer": True},
    "options": {"checks": ["duplicates", "numbers", "ai", "code"]},
    "students": {
        "a": {"name": "Nguyễn Văn An", "email": "an@school.test", "status": "turned_in",
              "submission_id": "sa", "notes": [], "files": ["Essay", "sum.py"]},
        "b": {"name": "Trần Bình", "email": "binh@school.test", "status": "late",
              "submission_id": "sb", "notes": [], "files": ["Essay"]},
        "c": {"name": "Lê Chi", "email": "chi@school.test", "status": "turned_in",
              "submission_id": "sc", "notes": ["bai.rar: RAR archives are not read."], "files": []},
        "d": {"name": "Phạm Dũng", "email": "dung@school.test", "status": "turned_in",
              "submission_id": "sd", "notes": [], "files": ["Essay"]},
    },
    "duplicates": [{"a": "a", "b": "b", "lexical": 0.62, "semantic": 0.98, "code": 1.0,
                    "llm": {"copied": True, "confidence": 90,
                            "evidence": ["Nguyễn Văn An and Trần Bình share identical sentences"]}}],
    "numbers": {"key": {"1": {"value": 42}, "2": {"value": 12.5}}, "results": {
        "a": {"results": [{"question": "1", "ok": True, "got": 42}, {"question": "2", "ok": True, "got": 12.5}],
              "correct": 2, "total": 2, "source": "pattern"},
        "b": {"results": [{"question": "1", "ok": False, "got": 41}, {"question": "2", "ok": True, "got": 12.5}],
              "correct": 1, "total": 2, "source": "pattern", "truncated": ["Student work"]},
        "d": {"results": [{"question": "1", "ok": True, "got": 42}, {"question": "2", "ok": True, "got": 12.5}],
              "correct": 2, "total": 2, "source": "pattern"}}},
    "rubric": None,
    "ai": {"b": {"heuristic": {"score": 65, "signals": ["Typical AI phrasing"]},
                 "llm": {"likelihood": 80, "reasons": ["generic tone, signed binh@school.test"]}}},
    "code": {"tests_defined": 2, "results": {
        "a": {"passed": 2, "total": 2, "llm": {"solves_task": True, "summary": "Adds numbers.", "issues": []}}}},
    "errors": [],
}


@pytest.fixture
def sample_report():
    """A checked assignment: a likely copy pair, an unreadable upload, one clean student."""
    report = copy.deepcopy(_SAMPLE)
    report["scores"] = _results.scores(report)
    return report
