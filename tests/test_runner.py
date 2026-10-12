import json

import pytest

from checker import runner


def test_parse_tests():
    tests = runner.parse_tests("--- input\n3 4\n--- expected\n7\n--- input\n1 1\n--- expected\n2\n--- expected\nhi")
    assert tests == [{"input": "3 4", "expected": "7"}, {"input": "1 1", "expected": "2"},
                     {"input": "", "expected": "hi"}]


def test_outputs_match():
    assert runner.outputs_match("7", "7\n")
    assert runner.outputs_match("1 2\n3", "1  2 3")
    assert runner.outputs_match("0.333333", "0.3333333333")
    assert not runner.outputs_match("7", "8")
    assert not runner.outputs_match("a b", "a")


def test_notebook_to_script():
    nb = {"cells": [{"cell_type": "markdown", "source": "# hi"},
                    {"cell_type": "code", "source": ["!pip install x\n", "print(1)\n"]},
                    {"cell_type": "code", "source": "%time\nprint(2)"}]}
    assert runner.notebook_to_script(json.dumps(nb)) == "print(1)\n\nprint(2)"


def test_language_of():
    assert runner.language_of("hw.PY") == "python"
    assert runner.language_of("nb.ipynb") == "python"
    assert runner.language_of("a.js") == "javascript"
    assert runner.language_of("essay.docx") is None


@pytest.fixture
def local_runner(monkeypatch):
    monkeypatch.setattr(runner, "CODE_RUNNER", "local")
    monkeypatch.setattr(runner, "TIMEOUT", 2)


def test_run_local(local_runner):
    src = "a, b = map(int, input().split())\nprint(a + b)\n"
    tests = runner.parse_tests("--- input\n3 4\n--- expected\n7\n--- input\n1 1\n--- expected\n3")
    r = runner.check(src, "python", tests, use_llm=False)
    assert (r["passed"], r["total"]) == (1, 2)


def test_run_local_timeout(local_runner):
    r = runner.run_one("while True: pass", "python")
    assert r["timed_out"]


def test_run_local_error(local_runner):
    r = runner.check("raise SystemExit(3)", "python", [], use_llm=False)
    assert r["passed"] == 0 and r["tests"][0]["exit_code"] == 3
