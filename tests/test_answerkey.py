"""Condensing a model answer sheet, and comparing a student against it."""
import json

import pytest

from checker import answerkey, llm, numbers

SHEET = """Đáp án đề kiểm tra giữa kỳ

Câu 1. Tính tổng hai số 20 và 22.
   Lời giải: 20 + 22 = 42. Vậy đáp án là 42.

Câu 2. Chu vi hình tròn bán kính 0,5 m.
   Lời giải: C = 2*pi*r = 3,14 m (chấp nhận sai số 0,01).

Câu 3. Giải thích định luật bảo toàn năng lượng.
   Lời giải: nêu được năng lượng không tự sinh ra hay mất đi, chỉ chuyển hoá;
   lấy được một ví dụ; có nêu đơn vị Joule.
"""

DISTILLED = {"questions": [
    {"label": "1", "answer": "42", "tol": None, "points": []},
    {"label": "2", "answer": "3,14", "tol": "0,01", "points": []},
    {"label": "3", "answer": "", "tol": None,
     "points": ["năng lượng không tự sinh ra hay mất đi", "có ví dụ", "đơn vị Joule"]},
]}


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the LLM with a canned reply and count the calls."""
    calls = []

    def chat_json(system, user, model=None, reply_tokens=None):
        calls.append(system)
        if "MODEL ANSWER SHEET" in system:
            return DISTILLED
        # grading: the student missed the example (point 3.2)
        return {"missing": ["3.2"], "note": "No example given."}

    monkeypatch.setattr(llm, "chat_json", chat_json)
    return calls


def test_markdown_round_trips_through_parse_key(fake_llm):
    """The condensed markdown IS a valid answer key: no second LLM call to compare."""
    key = answerkey.distill(SHEET)
    md = answerkey.to_markdown(key, "Giữa kỳ")

    parsed = numbers.parse_key(md)
    assert set(parsed) == {"1", "2"}                  # written answers are not numeric answers
    assert parsed["1"]["value"] == 42
    assert parsed["2"]["value"] == 3.14               # decimal comma
    assert parsed["2"]["tol"] == "0,01"
    assert numbers.is_close(3.14, 3.145, parsed["2"]["tol"])
    assert not numbers.is_close(3.14, 3.20, parsed["2"]["tol"])
    assert "- 3: năng lượng không tự sinh ra hay mất đi; có ví dụ; đơn vị Joule" in md


def test_cache_skips_the_llm_on_the_same_sheet(fake_llm, tmp_path):
    first = answerkey.load_or_build(SHEET, title="t", directory=str(tmp_path))
    assert first["cached"] is False and len(fake_llm) == 1

    second = answerkey.load_or_build(SHEET, title="t", directory=str(tmp_path))
    assert second["cached"] is True
    assert len(fake_llm) == 1                         # no second distillation
    assert second["markdown"] == first["markdown"]

    md_file = tmp_path / (first["path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1][:-5] + ".md")
    assert md_file.read_text(encoding="utf-8") == first["markdown"]

    # A different sheet is a different key.
    other = answerkey.load_or_build(SHEET + "\nCâu 4. ...", title="t", directory=str(tmp_path))
    assert other["cached"] is False and len(fake_llm) == 2


def test_grade_maps_missing_ids_back_to_their_question(fake_llm):
    questions = [q for q in answerkey.distill(SHEET)["questions"] if q["points"]]
    result = answerkey.grade("Năng lượng không tự sinh ra hay mất đi, đo bằng Joule.", questions)

    assert result["total"] == 3 and result["covered"] == 2
    q3 = result["results"][0]
    assert q3["question"] == "3"
    assert q3["missing"] == ["có ví dụ"]
    assert q3["score"] == 67
    assert result["note"] == "No example given."


def test_points_said_word_for_word_never_reach_the_llm(monkeypatch):
    """The cheap phrase match settles verbatim points, and is not fooled by word order."""
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: pytest.fail("LLM must not be called"))
    questions = [{"label": "3", "answer": "", "tol": None,
                  "points": ["năng lượng không tự sinh ra", "đơn vị Joule"]}]
    result = answerkey.grade(
        "Trả lời: năng lượng không tự sinh ra, và đơn vị Joule được dùng.", questions)
    assert result["asked"] == 0 and result["covered"] == 2
    assert result["results"][0]["missing"] == []


def test_phrase_match_respects_word_order(fake_llm):
    """Same words in a different order is not a match; the LLM decides instead."""
    questions = [{"label": "1", "answer": "", "tol": None, "points": ["sinh ra tự nhiên"]}]
    answerkey.grade("tự nhiên sinh ra", questions)
    assert len(fake_llm) == 1                         # fell through to the LLM


def test_pattern_only_needs_no_llm(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: pytest.fail("LLM must not be called"))
    key = answerkey.distill("1: 42\n2: 3.14 ± 0.01\n", use_llm=False)
    assert key["source"] == "pattern"
    assert [q["label"] for q in key["questions"]] == ["1", "2"]
    assert numbers.parse_key(answerkey.to_markdown(key))["2"]["tol"] == "0.01"


def test_no_llm_on_a_prose_sheet_warns_instead_of_inventing_answers(monkeypatch):
    """Patterns must not lift "22" out of "Tính tổng hai số 20 và 22." as the answer."""
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: pytest.fail("LLM must not be called"))
    key = answerkey.distill(SHEET, use_llm=False)
    assert key["questions"] == []
    assert any("--no-llm" in e for e in key["errors"])


def test_pattern_mode_keeps_short_answers_and_drops_question_text():
    assert answerkey._looks_like_an_answer("42")
    assert answerkey._looks_like_an_answer("3.14 ")
    assert answerkey._looks_like_an_answer("25%")
    assert answerkey._looks_like_an_answer("42 apples")
    assert not answerkey._looks_like_an_answer("Tính tổng hai số 20 và 22.")
    assert not answerkey._looks_like_an_answer("Chu vi hình tròn bán kính 0,5 m.")


def test_llm_failure_falls_back_to_patterns(monkeypatch):
    monkeypatch.setattr(llm, "chat_json",
                        lambda *a, **k: (_ for _ in ()).throw(llm.LLMError("ollama down")))
    key = answerkey.distill("1: 42\n")
    assert [q["label"] for q in key["questions"]] == ["1"]
    assert any("ollama down" in e for e in key["errors"])


def test_read_upload():
    """Format handling lives in checker.documents; this is the answer-sheet wiring."""
    assert answerkey.read_upload("key.md", b"1: 42\n") == "1: 42\n"
    nb = json.dumps({"cells": [{"cell_type": "code", "source": "print(42)"}]})
    assert answerkey.read_upload("key.ipynb", nb.encode()) == "print(42)"
    with pytest.raises(ValueError, match="Google Doc"):
        answerkey.read_upload("dapan.pdf", b"%PDF-1.4 binary")
    with pytest.raises(ValueError, match="could not be read"):
        answerkey.read_upload("dapan.txt", bytes([0xff, 0xfe, 0x00, 0x01]))
