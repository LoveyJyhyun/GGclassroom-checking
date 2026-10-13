"""Full pipeline against a fake Classroom and a fake Ollama server."""
import io
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from checker import answerkey, checks, llm, runner

ESSAY = ("Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose and oxygen. "
         "It happens in the chloroplasts, where chlorophyll absorbs mostly red and blue light. ") * 6


class FakeOllama(BaseHTTPRequestHandler):
    ASKED = []            # every (path, model) the checker requested

    def log_message(self, *args):
        pass

    def _send(self, out):
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"models": [
            {"name": "qwen2.5:7b", "size": 4_700_000_000,
             "details": {"parameter_size": "7.6B"}},
            {"name": "qwen3:14b", "size": 9_300_000_000,
             "details": {"parameter_size": "14.8B"}},
            {"name": "nomic-embed-text:latest", "size": 300_000_000,
             "details": {"parameter_size": "137M"}},
        ]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.ASKED.append((self.path, body.get("model")))
        if self.path == "/api/show":
            name = body.get("model", "")
            embedding = "embed" in name
            params = {"qwen3:14b": "14.8B", "qwen2.5:7b": "7.6B"}.get(name, "137M")
            self._send({
                "capabilities": ["embedding"] if embedding else ["completion"],
                "details": {"parameter_size": params},
                "model_info": {"x.context_length": 2048 if embedding else 32768},
            })
            return
        if self.path == "/api/embed":
            out = {"embeddings": [[1.0, float(len(t) % 7), 0.5] for t in body["input"]]}
        else:
            system = body["messages"][0]["content"]
            if "MODEL ANSWER SHEET" in system:
                content = {"questions": [
                    {"label": "1", "answer": "42", "tol": None, "points": []},
                    {"label": "2", "answer": "12.5", "tol": None, "points": []},
                    {"label": "3", "answer": "", "tol": None,
                     "points": ["chlorophyll absorbs red and blue light", "names glucose"]},
                ]}
            elif "REQUIRED POINTS" in system:
                content = {"missing": ["3.2"], "note": "Glucose not named."}
            elif "FINAL numeric answers" in system:
                content = {"answers": {"2": "12.5"}}
            elif "copied" in system:
                content = {"copied": True, "confidence": 90, "evidence": ["identical sentences"]}
            elif "AI assistant" in system:
                content = {"likelihood": 35, "reasons": ["generic tone"]}
            else:
                content = {"solves_task": True, "issues": [], "summary": "Adds two numbers."}
            out = {"message": {"content": json.dumps(content)}}
        self._send(out)


@pytest.fixture
def fake_ollama(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(llm, "_DESCRIBED", {})      # per-model metadata is cached
    FakeOllama.ASKED.clear()
    yield
    server.shutdown()


def run_check(client, data, timeout=120):
    """POST the check form, wait for the background job, return (status, report HTML).

    This is the real flow the browser follows: the POST only starts a thread and
    redirects to the progress page.
    """
    resp = client.post("/course/c1/work/w1/check", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 302, resp.data[:400]
    job_path = urllib.parse.urlparse(resp.headers["Location"]).path
    assert job_path.startswith("/job/"), job_path

    seen_totals, deadline = [], time.time() + timeout
    status = None
    while time.time() < deadline:
        status = client.get(job_path + "/status").get_json()
        if status["total"]:
            seen_totals.append((status["done"], status["total"]))
        if status["state"] != "running":
            break
        time.sleep(0.02)
    assert status and status["state"] == "done", status
    assert status["done"] == status["total"], status      # the bar reaches the end
    assert all(d <= t for d, t in seen_totals)            # and never overshoots

    page = client.get("/report/" + status["report_file"])
    assert page.status_code == 200
    return status, page.data.decode()


def _student(uid, name):
    return {"userId": uid, "profile": {"name": {"fullName": name}, "emailAddress": f"{uid}@school.test"}}


def _sub(uid, *file_ids, state="TURNED_IN"):
    return {"id": "s" + uid, "userId": uid, "courseWorkId": "w1", "state": state,
            "assignmentSubmission": {"attachments": [
                {"driveFile": {"id": f, "title": f, "alternateLink": "#"}} for f in file_ids]}}


FILES = {
    "doc-a": ("Essay", "application/vnd.google-apps.document", ESSAY + "\n1: 42\n2: twelve and a half"),
    "doc-b": ("Essay", "application/vnd.google-apps.document", ESSAY + " Me too.\n1: 41\n2: 12.5"),
    "doc-c": ("Essay", "application/vnd.google-apps.document", "Short note. 1: 42"),
    "code-a": ("sum.py", "text/x-python", "a, b = map(int, input().split())\nprint(a + b)\n"),
    "code-b": ("main.py", "text/x-python", "x, y = map(int, input().split())\nprint(x + y)\n"),
    "code-c": ("nb.ipynb", "application/vnd.google.colaboratory",
               json.dumps({"cells": [{"cell_type": "code", "source": "print(8)"}]})),
}


class FakeClassroom:
    def one_course_work(self, course_id, work_id):
        return {"id": work_id, "title": "Lab 1", "description": "Add two numbers.", "alternateLink": "#"}

    def course(self, course_id):
        return {"id": course_id, "name": "Science"}

    def students(self, course_id):
        return [_student("a", "An"), _student("b", "Binh"), _student("c", "Chi"), _student("d", "Dung")]

    def submissions(self, course_id, work_id="-"):
        return [_sub("a", "doc-a", "code-a"), _sub("b", "doc-b", "code-b"),
                _sub("c", "doc-c", "code-c"), _sub("d", state="CREATED")]

    def file_content(self, file_id):
        name, mime, text = FILES[file_id]
        return {"name": name, "mime": mime, "text": text, "note": ""}


def test_full_report(fake_ollama, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "CODE_RUNNER", "local")
    opts = checks.Options(answer_key="1: 42\n2: 12.5",
                          tests="--- input\n3 4\n--- expected\n7")
    report = checks.run(FakeClassroom(), "c1", "w1", opts)

    assert set(report["students"]) == {"a", "b", "c"}          # d has not turned in
    assert report["errors"] == []

    top = report["duplicates"][0]
    assert {top["a"], top["b"]} == {"a", "b"}
    assert top["code"] == 1.0 and top["llm"]["copied"]

    nums = report["numbers"]["results"]
    assert nums["a"]["correct"] == 2 and nums["a"]["source"] == "pattern + llm"
    assert nums["b"]["correct"] == 1
    assert nums["c"]["correct"] == 1

    assert report["ai"]["a"]["llm"]["likelihood"] == 35
    assert report["ai"]["c"]["heuristic"]["score"] is None      # too short

    code = report["code"]["results"]
    assert code["a"]["passed"] == 1 and code["b"]["passed"] == 1
    assert code["c"]["passed"] == 0                             # notebook output is wrong
    assert code["a"]["llm"]["summary"] == "Adds two numbers."

    path = checks.save(report, str(tmp_path))
    assert json.load(open(path))["work"]["title"] == "Lab 1"

    # Report page renders.
    import app as webapp
    with webapp.app.test_request_context():
        html = webapp.render_template("report.html", course={"id": "c1", "name": "Science"},
                                      work={"id": "w1", "title": "Lab 1"}, report=report,
                                      report_file="r.json", labels={})
    assert "Possible copying" in html and "likely copied" in html and "2/2" in html


class FakeWebClassroom(FakeClassroom):
    def courses(self):
        return [{"id": "c1", "name": "Science", "section": "10A"}]

    def course_work(self, course_id):
        return [self.one_course_work(course_id, "w1")]

    def submission(self, course_id, work_id, sub_id):
        return next(s for s in self.submissions(course_id) if s["id"] == sub_id)

    def student(self, course_id, user_id):
        return next(s for s in self.students(course_id) if s["userId"] == user_id)


def test_web_pages(fake_ollama, monkeypatch, tmp_path):
    import app as webapp
    monkeypatch.setattr(runner, "CODE_RUNNER", "local")
    monkeypatch.setattr(checks, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(webapp, "_credentials", lambda: object())
    monkeypatch.setattr(webapp, "Classroom", lambda creds: FakeWebClassroom())
    client = webapp.app.test_client()
    with client.session_transaction() as s:
        s["credentials"] = {"token": "x"}

    assert b"Science" in client.get("/").data
    page = client.get("/course/c1").data.decode()
    assert "Lab 1" in page and "Missing" in page
    csv = client.get("/course/c1/export.csv").data.decode()
    assert csv.splitlines()[0].startswith("Student,Email,Missing,Lab 1")
    assert "Turned in" in client.get("/course/c1/work/w1").data.decode()
    assert "a, b = map(int" in client.get("/course/c1/work/w1/submission/sa").data.decode()
    assert "Ollama" in client.get("/course/c1/work/w1/check").data.decode()

    status, html = run_check(client, {
        "duplicates": "on", "numbers": "on", "ai": "on", "code": "on", "use_llm": "on",
        "answer_key": "1: 42\n2: 12.5", "tests": "--- input\n3 4\n--- expected\n7"})
    assert "likely copied" in html and "Number answers" in html
    assert "Final score" in html

    report_file = status["report_file"]
    assert client.get("/reports/" + report_file).status_code == 200
    assert client.get("/reports/../app.py").status_code == 404
    assert client.get("/report/..%2fapp.py").status_code == 404

    csv_text = client.get("/report/" + report_file + "/csv").data.decode("utf-8-sig")
    header, *rows = csv_text.strip().splitlines()
    assert header.startswith("Student,Email,Status,Score %")
    assert len(rows) == 3 and rows[0].startswith("An,")


MODEL_SHEET = """Model answers for Lab 1

Question 1. Add 20 and 22.
   Answer: 42

Question 2. Half of twenty-five.
   Answer: 12.5

Question 3. Describe photosynthesis.
   Answer: chlorophyll absorbs red and blue light; names glucose as the product.
"""


def test_model_answer_sheet_fills_the_key_and_drives_the_rubric(fake_ollama, monkeypatch, tmp_path):
    monkeypatch.setattr(answerkey, "KEYS_DIR", str(tmp_path))
    opts = checks.Options(checks=("numbers", "rubric"), model_answer=MODEL_SHEET)
    report = checks.run(FakeClassroom(), "c1", "w1", opts)
    assert report["errors"] == []

    # The sheet was condensed once into a key the number check reads directly.
    assert report["answer_key"]["questions"] == 3
    assert report["answer_key"]["cached"] is False
    assert "1: 42" in report["answer_key"]["markdown"]
    nums = report["numbers"]["results"]
    assert nums["a"]["correct"] == 2 and nums["b"]["correct"] == 1

    # Only question 3 has written points, so only it is compared point by point.
    rub = report["rubric"]
    assert [q["label"] for q in rub["questions"]] == ["3"]
    a = rub["results"]["a"]
    assert a["total"] == 2 and a["covered"] == 1
    assert a["results"][0]["missing"] == ["names glucose"]
    # The essay says "absorbs *mostly* red and blue light", so the cheap phrase match
    # declines and both points go to the LLM. test_answerkey covers the free path.
    assert a["asked"] == 2

    # Re-running with the same sheet reuses the cached key.
    again = checks.run(FakeClassroom(), "c1", "w1", opts)
    assert again["answer_key"]["cached"] is True


def test_check_form_accepts_an_uploaded_sheet(fake_ollama, monkeypatch, tmp_path):
    import app as webapp
    monkeypatch.setattr(checks, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(answerkey, "KEYS_DIR", str(tmp_path / "keys"))
    monkeypatch.setattr(webapp, "_credentials", lambda: object())
    monkeypatch.setattr(webapp, "Classroom", lambda creds: FakeWebClassroom())
    client = webapp.app.test_client()
    with client.session_transaction() as s:
        s["credentials"] = {"token": "x"}

    status, html = run_check(client, {
        "numbers": "on", "rubric": "on", "use_llm": "on",
        "model_answer_file": (io.BytesIO(MODEL_SHEET.encode()), "dapan.md")})
    assert "Condensed answer key" in html and "1: 42" in html
    assert "Compared with the model answer" in html and "names glucose" in html

    csv_text = client.get("/report/" + status["report_file"] + "/csv").data.decode("utf-8-sig")
    assert "Q1,Q2,Q3" in csv_text.splitlines()[0]

    # A binary sheet comes back as an error on the form, not a 500.
    bad = client.post("/course/c1/work/w1/check", data={
        "numbers": "on", "model_answer_file": (io.BytesIO(b"%PDF-1.4"), "dapan.pdf"),
    }, content_type="multipart/form-data")
    assert bad.status_code == 400
    assert "PDF text is not read" in bad.data.decode()


def test_phases_run_in_the_documented_order(fake_ollama, monkeypatch, tmp_path):
    """1. copying, 2. answers, 3. AI text, 4. code - and the bar lands on 100%."""
    monkeypatch.setattr(runner, "CODE_RUNNER", "local")
    monkeypatch.setattr(answerkey, "KEYS_DIR", str(tmp_path))

    seen = []
    pr = checks.Progress(lambda phase, msg, done, total: seen.append((phase, done, total)))
    opts = checks.Options(model_answer=MODEL_SHEET,
                          tests="--- input" + chr(10) + "3 4" + chr(10) + "--- expected" + chr(10) + "7")
    report = checks.run(FakeClassroom(), "c1", "w1", opts, progress=pr)

    phases = [p for i, (p, _, _) in enumerate(seen) if i == 0 or p != seen[i - 1][0]]
    assert phases == [
        "Reading submissions",
        "Condensing the model answer",
        "1. Copying between students",
        "2. Answers against the key",
        "2. Written answers against the key",
        "3. AI-written text",
        "4. Running code",
    ], phases

    assert seen[-1][1] == seen[-1][2]                        # finishes at 100%
    assert all(d <= t for _, d, t in seen if t)              # never overshoots
    assert all(b[1] >= a[1] for a, b in zip(seen, seen[1:]))  # never goes backwards

    # Marks come from the answer sheet: An got both numbers and 1 of 2 written points.
    assert report["scores"]["a"]["per_question"] == {"1": 1.0, "2": 1.0, "3": 0.5}
    assert report["scores"]["a"]["percent"] == 83.3


def test_bar_completes_when_rubric_is_on_but_no_sheet_was_given(fake_ollama, monkeypatch):
    """The rubric phase is skipped without a sheet, so its steps must not be planned."""
    monkeypatch.setattr(runner, "CODE_RUNNER", "local")
    seen = []
    pr = checks.Progress(lambda phase, msg, done, total: seen.append((done, total)))
    checks.run(FakeClassroom(), "c1", "w1",
               checks.Options(checks=("duplicates", "numbers", "rubric", "ai", "code"),
                              answer_key="1: 42", tests=""), progress=pr)
    assert seen[-1][0] == seen[-1][1], seen[-5:]


def test_a_sheet_with_no_answer_check_selected_says_why_there_is_no_score(
        fake_ollama, monkeypatch, tmp_path):
    monkeypatch.setattr(answerkey, "KEYS_DIR", str(tmp_path))
    report = checks.run(FakeClassroom(), "c1", "w1",
                        checks.Options(checks=("duplicates",), model_answer=MODEL_SHEET))
    assert not any(report["scores"].values())
    assert any("No score was worked out" in e for e in report["errors"])


def test_written_only_questions_get_a_column_in_the_score_table(
        fake_ollama, monkeypatch, tmp_path):
    """Question 3 is marked by points only; the CSV had it but the table did not."""
    import app as webapp
    monkeypatch.setattr(answerkey, "KEYS_DIR", str(tmp_path))
    report = checks.run(FakeClassroom(), "c1", "w1",
                        checks.Options(checks=("numbers", "rubric"), model_answer=MODEL_SHEET))
    assert report["question_labels"] == ["1", "2", "3"]
    with webapp.app.test_request_context():
        html = webapp.render_template("report.html", course=report["course"],
                                      work=report["work"], report=report,
                                      report_file="x.json", labels={})
    assert ">Q3<" in html


def test_the_chosen_model_is_the_one_actually_called(fake_ollama, monkeypatch, tmp_path):
    """Picking a model must reach every LLM call, not just the report header."""
    monkeypatch.setattr(runner, "CODE_RUNNER", "local")
    monkeypatch.setattr(answerkey, "KEYS_DIR", str(tmp_path))
    opts = checks.Options(checks=("duplicates", "numbers", "ai"),
                          model="qwen3:14b", embed_model="nomic-embed-text:latest",
                          answer_key="1: 42")
    report = checks.run(FakeClassroom(), "c1", "w1", opts)

    assert report["options"]["model"] == "qwen3:14b"
    assert report["options"]["embed_model"] == "nomic-embed-text:latest"

    chat_models = {m for path, m in FakeOllama.ASKED if path == "/api/chat"}
    embed_models = {m for path, m in FakeOllama.ASKED if path == "/api/embed"}
    assert chat_models == {"qwen3:14b"}, chat_models
    assert embed_models == {"nomic-embed-text:latest"}, embed_models


def test_an_empty_choice_keeps_the_configured_default(fake_ollama, monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_MODEL", "qwen2.5:7b")
    checks.run(FakeClassroom(), "c1", "w1",
               checks.Options(checks=("ai",), model="", embed_model=""))
    assert {m for path, m in FakeOllama.ASKED if path == "/api/chat"} == {"qwen2.5:7b"}


def test_the_answer_key_cache_is_per_model(fake_ollama, monkeypatch, tmp_path):
    """A different model condenses differently, so it must not reuse the cached key."""
    monkeypatch.setattr(answerkey, "KEYS_DIR", str(tmp_path))
    first = checks.run(FakeClassroom(), "c1", "w1",
                       checks.Options(checks=("numbers",), model="qwen2.5:7b",
                                      model_answer=MODEL_SHEET))
    again = checks.run(FakeClassroom(), "c1", "w1",
                       checks.Options(checks=("numbers",), model="qwen2.5:7b",
                                      model_answer=MODEL_SHEET))
    other = checks.run(FakeClassroom(), "c1", "w1",
                       checks.Options(checks=("numbers",), model="qwen3:14b",
                                      model_answer=MODEL_SHEET))
    assert first["answer_key"]["cached"] is False
    assert again["answer_key"]["cached"] is True      # same model, reused
    assert other["answer_key"]["cached"] is False     # different model, rebuilt


def test_the_form_lists_the_installed_models(fake_ollama, monkeypatch):
    import app as webapp
    monkeypatch.setattr(webapp, "_credentials", lambda: object())
    monkeypatch.setattr(webapp, "Classroom", lambda creds: FakeWebClassroom())
    client = webapp.app.test_client()
    with client.session_transaction() as s:
        s["credentials"] = {"token": "x"}

    page = client.get("/course/c1/work/w1/check").data.decode()
    assert 'name="model"' in page and 'name="embed_model"' in page
    assert "qwen3:14b" in page and "14.8B" in page
    # The embedding model belongs in its own list, not the chat one.
    chat_block = page.split('name="model"')[1].split("</select>")[0]
    embed_block = page.split('name="embed_model"')[1].split("</select>")[0]
    assert "nomic-embed-text" not in chat_block
    assert "nomic-embed-text" in embed_block and "qwen3:14b" not in embed_block
