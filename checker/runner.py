"""Run student code against teacher test cases.

Student code is untrusted. The default runner is Docker with no network,
a read-only filesystem and memory/CPU/process limits. Running directly on
the host (CODE_RUNNER=local) only applies rlimits and should be used only
when you trust the code or are already inside a disposable VM.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

from . import llm
from .numbers import parse_number

CODE_RUNNER = os.environ.get("CODE_RUNNER", "docker")  # "docker" or "local"
TIMEOUT = float(os.environ.get("CODE_TIMEOUT", "10"))
MEMORY_MB = int(os.environ.get("CODE_MEMORY_MB", "256"))
MAX_OUTPUT = 20_000
# When this app runs inside Docker and starts sibling containers through the host's
# Docker socket, the code folder must exist at the same path on the host.
CODE_WORK_DIR = os.environ.get("CODE_WORK_DIR") or None

LANGUAGES = {
    "python": {"ext": ".py", "image": os.environ.get("PYTHON_IMAGE", "python:3.12-slim"),
               "docker": ["python", "-I", "main.py"], "local": [sys.executable, "-I", "main.py"]},
    "javascript": {"ext": ".js", "image": os.environ.get("NODE_IMAGE", "node:20-slim"),
                   "docker": ["node", "main.js"], "local": ["node", "main.js"]},
}
EXTENSIONS = {".py": "python", ".ipynb": "python", ".js": "javascript"}


def language_of(filename: str, mime: str = "") -> str | None:
    name = filename.lower()
    for ext, lang in EXTENSIONS.items():
        if name.endswith(ext):
            return lang
    if "python" in mime or "colaboratory" in mime:
        return "python"
    if "javascript" in mime:
        return "javascript"
    return None


def notebook_to_script(raw: str) -> str:
    """Join the code cells of a Jupyter/Colab notebook, dropping shell/magic lines."""
    nb = json.loads(raw)
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def parse_tests(text: str) -> list[dict]:
    """Parse test cases written as:

        --- input
        3 4
        --- expected
        7
        --- input
        ...

    A test without '--- input' runs with empty stdin. An '--- expected' block
    with no input block before it is allowed (e.g. a program that takes no input).
    """
    tests, current, section, buf = [], None, None, []

    def flush():
        if current is not None and section:
            current[section] = "\n".join(buf).strip("\n")

    for line in text.splitlines():
        m = re.fullmatch(r"\s*-{3,}\s*(input|expected|output)\s*-*\s*", line, re.IGNORECASE)
        if m:
            flush()
            name = "expected" if m.group(1).lower() in ("expected", "output") else "input"
            if name == "input" or section != "input":
                current = {"input": "", "expected": None}
                tests.append(current)
            section, buf = name, []
        else:
            buf.append(line)
    flush()
    return tests


def outputs_match(expected: str, got: str, rel_tol: float = 1e-6) -> bool:
    """Compare ignoring whitespace layout; numbers compared with tolerance."""
    exp_tokens, got_tokens = expected.split(), got.split()
    if exp_tokens == got_tokens:
        return True
    if len(exp_tokens) != len(got_tokens):
        return False
    for e, g in zip(exp_tokens, got_tokens):
        if e == g:
            continue
        ev, gv = parse_number(e), parse_number(g)
        if ev is None or gv is None or abs(ev - gv) > max(rel_tol * abs(ev), 1e-9):
            return False
    return True


def _limits():  # pragma: no cover - runs in the child process
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (int(TIMEOUT) + 1, int(TIMEOUT) + 1))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_MB * 2**20, MEMORY_MB * 2**20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 2**20, 10 * 2**20))
    os.setsid()


def runner_available() -> tuple[bool, str]:
    if CODE_RUNNER == "local":
        return True, "local (no isolation)"
    if shutil.which("docker") is None:
        return False, "Docker not found. Install Docker, or set CODE_RUNNER=local to run without isolation."
    try:
        ok = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                            capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    if not ok:
        return False, "Docker is installed but the daemon is not running. Start Docker Desktop / dockerd."
    return True, "docker"


def run_one(source: str, language: str, stdin: str = "") -> dict:
    spec = LANGUAGES[language]
    if CODE_WORK_DIR:
        os.makedirs(CODE_WORK_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ggc-run-", dir=CODE_WORK_DIR) as work:
        path = os.path.join(work, "main" + spec["ext"])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        os.chmod(work, 0o755)
        os.chmod(path, 0o644)

        name = None
        if CODE_RUNNER == "local":
            cmd, kwargs = spec["local"], {"cwd": work, "preexec_fn": _limits,
                                          "env": {"PATH": os.environ.get("PATH", ""), "HOME": work}}
        else:
            name = f"ggc-{uuid.uuid4().hex[:12]}"
            cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none",
                   "--memory", f"{MEMORY_MB}m", "--cpus", "1", "--pids-limit", "64",
                   "--read-only", "--tmpfs", "/tmp:size=16m", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--user", "65534:65534",
                   "-v", f"{work}:/work:ro", "-w", "/work", spec["image"],
                   "timeout", "-s", "KILL", str(max(1, round(TIMEOUT))), *spec["docker"]]
            kwargs = {}
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                                  timeout=TIMEOUT + (30 if name else 0), **kwargs)
        except subprocess.TimeoutExpired as e:
            if name:
                subprocess.run(["docker", "kill", name], capture_output=True)
            return {"stdout": (e.stdout or "")[:MAX_OUTPUT] if isinstance(e.stdout, str) else "",
                    "stderr": "Timed out.", "exit_code": None, "timed_out": True}
        except FileNotFoundError as e:
            return {"stdout": "", "stderr": f"Runner not available: {e}", "exit_code": None,
                    "timed_out": False}
    # 137 = killed by the in-container `timeout -s KILL` (or the CPU rlimit locally).
    timed_out = proc.returncode in (137, -9, -24)
    return {"stdout": proc.stdout[:MAX_OUTPUT],
            "stderr": "Timed out." if timed_out and not proc.stderr else proc.stderr[-MAX_OUTPUT:],
            "exit_code": proc.returncode, "timed_out": timed_out}


REVIEW_SYSTEM = (
    "You review a student's program for a teacher. Using the assignment, the code and "
    "the test results, say whether the code solves the task and list concrete problems "
    "(bugs, hard-coded outputs, missing requirements). Do not rewrite the code. "
    'Reply with JSON only: {"solves_task": true|false, "issues": ["..."], "summary": "one sentence"}'
)


def llm_review(source: str, assignment: str, results: list[dict]) -> dict:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"Test {i}: {'PASS' if r['passed'] else 'FAIL'}; "
                     f"stdin={r['input'][:200]!r}; expected={str(r['expected'])[:200]!r}; "
                     f"got={r['stdout'][:200]!r}; stderr={r['stderr'][-300:]!r}")
    prompt = (f"Assignment:\n{assignment or '(not provided)'}\n\n"
              f"Code:\n<<<\n{source}\n>>>\n\nTest results:\n" + ("\n".join(lines) or "(no tests)"))
    data = llm.chat_json(REVIEW_SYSTEM, prompt)
    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    return {"solves_task": bool(data.get("solves_task")), "issues": [str(i) for i in issues][:8],
            "summary": str(data.get("summary", ""))}


def check(source: str, language: str, tests: list[dict], assignment: str = "",
          use_llm: bool = True) -> dict:
    available, note = runner_available()
    if not available:
        return {"error": note, "tests": [], "passed": 0, "total": len(tests), "llm": None}
    cases = tests or [{"input": "", "expected": None}]
    results = []
    for case in cases:
        run = run_one(source, language, case["input"])
        if case["expected"] is None:
            passed = run["exit_code"] == 0
        else:
            passed = run["exit_code"] == 0 and outputs_match(case["expected"], run["stdout"])
        results.append({**case, **run, "passed": passed})
    report = {"error": None, "runner": note, "tests": results,
              "passed": sum(r["passed"] for r in results), "total": len(results), "llm": None}
    if use_llm:
        try:
            report["llm"] = llm_review(source, assignment, results)
        except llm.LLMError as e:
            report["error"] = str(e)
    return report
