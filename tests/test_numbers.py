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
    monkeypatch.setattr(nb, "extract_with_llm", lambda text, labels: {"2": 7.0})
    key = nb.parse_key("1: 5\n2: 7")
    r = nb.check("1: 5\nSecond one, the answer is seven: 7", key)
    assert r["correct"] == 2 and r["source"] == "pattern + llm"
