from checker import triage


def test_codes_are_stable_and_in_name_order(sample_report):
    assert triage.codes(sample_report) == {"c": "S01", "a": "S02", "d": "S03", "b": "S04"}


def test_tiers_follow_the_strongest_signal(sample_report):
    b = triage.brief(sample_report, "r.json")
    tiers = {p["code"]: p["tier"] for p in b["students"]}
    assert tiers == {"S02": "act", "S04": "act",          # likely copied
                     "S01": "act",                        # nothing readable
                     "S03": "ok"}
    order = [p["code"] for p in b["students"]]
    assert order[-1] == "S03"                             # concerns first
    assert b["stats"]["copy_pairs"] == 1 and b["stats"]["ai_flagged"] == 1


def test_ai_signals_alone_never_reach_action(sample_report):
    sample_report["duplicates"] = []
    for r in sample_report["numbers"]["results"].values():
        r.pop("truncated", None)
    b = triage.brief(sample_report)
    s04 = next(p for p in b["students"] if p["code"] == "S04")
    assert s04["tier"] == "review"
    assert {s["kind"] for s in s04["signals"]} == {"ai", "late"}


def test_a_confident_not_copied_review_clears_the_pair(sample_report):
    sample_report["duplicates"][0].update(lexical=0.3, llm={"copied": False, "confidence": 85, "evidence": []})
    copy = [s for p in triage.brief(sample_report)["students"] for s in p["signals"] if s["kind"] == "copying"]
    assert copy and all(s["level"] == "info" for s in copy)


def test_the_brief_holds_no_names_or_emails(sample_report):
    import re
    text = triage.brief_markdown(triage.brief(sample_report))
    for s in sample_report["students"].values():
        assert s["name"] not in text and s["email"] not in text
        local = s["email"].split("@")[0]
        if len(local) >= 3:                       # 2-letter tokens would hit ordinary words
            assert not re.search(rf"\b{local}\b", text)
    assert '"S02 and S04 share identical sentences"' in text      # evidence kept, names scrubbed
    assert "signed S04" in text
    assert text.startswith("# Triage brief: Lab 1")              # text, not JSON, for Dify


def test_unreadable_work_is_unmarked_not_low(sample_report):
    b = triage.brief(sample_report)
    s01 = next(p for p in b["students"] if p["code"] == "S01")
    assert s01["score"] is None
    assert [s["kind"] for s in s01["signals"]] == ["unreadable"]


def test_student_detail(sample_report):
    d = triage.student_detail(sample_report, "s04")
    assert d["code"] == "S04" and d["score"]["percent"] == 50.0
    assert d["numeric"] == [{"question": "1", "ok": False}, {"question": "2", "ok": True}]
    assert triage.student_detail(sample_report, "S99") is None


def test_codes_parsed_and_names_restored(sample_report):
    assert triage.parse_codes('["S04", "s02", "S99", "S04", "S02"]', sample_report) == ["S04", "S02"]
    assert triage.with_names("Hold S04.", sample_report) == "Hold Trần Bình (S04)."


def test_triage_state_is_stored_next_to_the_report(tmp_path, sample_report):
    path = str(tmp_path / "20261013-130000_Lab-1_w1.json")
    assert triage.load_triage(path) is None
    triage.write_triage(path, state="running")
    triage.write_triage(path, state="done", text="ok", hold=["S04"])
    stored = triage.load_triage(path)
    assert stored["state"] == "done" and stored["hold"] == ["S04"] and stored["updated_at"]
    assert (tmp_path / "agent" / "20261013-130000_Lab-1_w1.triage.json").exists()
