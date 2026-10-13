from checker import numbers as nb


def test_parse_number():
    assert nb.parse_number("1,234.5") == 1234.5
    assert nb.parse_number("3,14") == 3.14
    assert nb.parse_number("3/4") == 0.75
    assert nb.parse_number("25%") == 0.25
    assert nb.parse_number("−2") == -2
    assert nb.parse_number("abc") is None


def test_parse_key():
    key = nb.parse_key("# key\n1: 42\n2) 3.14 ± 0.01\nQ3 = 25%\nCâu 4: 1/3 ± 1%\nnot a line")
    assert key["1"]["value"] == 42
    assert key["2"]["tol"] == "0.01"
    assert key["3"]["value"] == 0.25
    assert abs(key["4"]["value"] - 1 / 3) < 1e-12 and key["4"]["tol"] == "1%"


def test_check_without_llm():
    key = nb.parse_key("1: 42\n2: 3.14 ± 0.01\n3: 10")
    text = "Question 1: after working, x = 42\n2) pi is about 3.141\n3: I got 11"
    r = nb.check(text, key, use_llm=False)
    assert [x["ok"] for x in r["results"]] == [True, True, False]
    assert r["correct"] == 2


def test_check_uses_llm_for_missing(monkeypatch):
    monkeypatch.setattr(nb, "extract_with_llm", lambda text, labels: ({"2": 7.0}, []))
    key = nb.parse_key("1: 5\n2: 7")
    r = nb.check("1: 5\nSecond one, the answer is seven: 7", key)
    assert r["correct"] == 2 and r["source"] == "pattern + llm"


def test_tolerance_written_with_a_decimal_comma():
    """"2: 3,14 ± 0,01" must grade against 3.14, not against the tolerance."""
    key = nb.parse_key("2: 3,14 ± 0,01")
    assert key["2"]["value"] == 3.14 and key["2"]["tol"] == "0,01"
    assert nb.is_close(3.14, 3.145, key["2"]["tol"])
    assert not nb.is_close(3.14, 3.2, key["2"]["tol"])


def test_unreadable_tolerance_does_not_crash():
    assert nb.is_close(42, 42, "oops")
    assert not nb.is_close(42, 50, "oops")


def test_tolerance_written_in_words_or_brackets():
    """Vietnamese sheets write "(sai so 0,01)", not "± 0,01"."""
    for line, value, tol in [
        ("2: 3,14 (sai số 0,01)", 3.14, "0,01"),
        ("3: 2.5 (tol 0.1)", 2.5, "0.1"),
        ("4: 1/3 +- 1%", 1 / 3, "1%"),
        ("5: 42", 42, None),
        ("6: 3,14 (sai so 0,01)", 3.14, "0,01"),      # no diacritics
        ("7: 3,14 (SAI SO 0,01)", 3.14, "0,01"),
    ]:
        key = nb.parse_key(line)
        label = next(iter(key))
        assert abs(key[label]["value"] - value) < 1e-9, (line, key)
        assert key[label]["tol"] == tol, (line, key)

    # Without this, the tolerance itself was graded as the answer.
    key = nb.parse_key("2: 3,14 (sai số 0,01)")
    assert nb.is_close(key["2"]["value"], 3.145, key["2"]["tol"])
    assert not nb.is_close(key["2"]["value"], 0.01, key["2"]["tol"])
