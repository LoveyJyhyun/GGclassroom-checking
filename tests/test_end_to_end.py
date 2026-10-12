"""Full pipeline against a fake Classroom and a fake Ollama server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from checker import checks, llm, runner

ESSAY = ("Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose and oxygen. "
         "It happens in the chloroplasts, where chlorophyll absorbs mostly red and blue light. ") * 6


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":
            out = {"embeddings": [[1.0, float(len(t) % 7), 0.5] for t in body["input"]]}
        else:
            system = body["messages"][0]["content"]
            if "FINAL numeric answers" in system:
                content = {"answers": {"2": "12.5"}}
            elif "copied" in system:
                content = {"copied": True, "confidence": 90, "evidence": ["identical sentences"]}
            elif "AI assistant" in system:
                content = {"likelihood": 35, "reasons": ["generic tone"]}
            else:
                content = {"solves_task": True, "issues": [], "summary": "Adds two numbers."}
            out = {"message": {"content": json.dumps(content)}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_ollama(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
    yield
    server.shutdown()


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

    resp = client.post("/course/c1/work/w1/check", data={
        "duplicates": "on", "numbers": "on", "ai": "on", "code": "on", "use_llm": "on",
        "answer_key": "1: 42\n2: 12.5", "tests": "--- input\n3 4\n--- expected\n7"})
    html = resp.data.decode()
    assert resp.status_code == 200 and "likely copied" in html and "Number answers" in html
    report_file = html.split("/reports/")[1].split('"')[0]
    assert client.get("/reports/" + report_file).status_code == 200
    assert client.get("/reports/../app.py").status_code == 404
