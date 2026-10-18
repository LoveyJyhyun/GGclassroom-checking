"""The agent side: MCP server, JSON tools, triage via Dify, and approving draft grades."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import app as webapp
from test_end_to_end import fake_ollama  # noqa: F401  (fixture)
from checker import agent_tools, checks, dify, grades, mcp, triage
from checker.classroom import COURSEWORK_WRITE

TOKEN = "test-token"
NEW = "20261013-130000_Lab-1_w1.json"
OLD = "20261001-090000_Lab-0_w0.json"


@pytest.fixture
def reports(tmp_path, monkeypatch, sample_report):
    monkeypatch.setattr(checks, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(webapp, "AGENT_TOKEN", TOKEN)
    old = json.loads(json.dumps(sample_report))
    old["work"]["title"] = "Lab 0"
    for name, report in ((NEW, sample_report), (OLD, old)):
        (tmp_path / name).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "20261013-130000_Lab-1_w1.csv").write_text("not a report")
    return tmp_path


@pytest.fixture
def client():
    return webapp.app.test_client()


def rpc(client, method, params=None, msg_id=1, token=TOKEN):
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if msg_id is not None:
        body["id"] = msg_id
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"}
    return client.post("/agent/mcp", json=body, headers=headers)


def call(client, name, **arguments):
    resp = rpc(client, "tools/call", {"name": name, "arguments": arguments})
    assert resp.status_code == 200, resp.data
    return resp.get_json()["result"]


# --- MCP protocol ------------------------------------------------------------

def test_agent_endpoints_are_off_without_a_token(client, monkeypatch):
    monkeypatch.setattr(webapp, "AGENT_TOKEN", "")
    assert rpc(client, "ping").status_code == 404
    assert client.get("/agent/tools").status_code == 404


def test_token_is_required(reports, client):
    assert rpc(client, "ping", token="wrong").status_code == 401
    assert client.post("/agent/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
    # Never in the URL, where it would be logged.
    assert client.post(f"/agent/mcp?key={TOKEN}",
                       json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
    assert rpc(client, "ping").status_code == 200


def test_initialize_negotiates_the_version(reports, client):
    resp = rpc(client, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                      "clientInfo": {"name": "t", "version": "1"}})
    result = resp.get_json()["result"]
    assert result["protocolVersion"] == "2025-03-26"
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "ggclassroom-checker"
    assert resp.headers["Mcp-Session-Id"]
    unknown = rpc(client, "initialize", {"protocolVersion": "1999-01-01"}).get_json()["result"]
    assert unknown["protocolVersion"] == mcp.LATEST


def test_notifications_get_no_reply_and_there_is_no_stream(reports, client):
    resp = rpc(client, "notifications/initialized", msg_id=None)
    assert resp.status_code == 202 and resp.data == b""
    assert client.get("/agent/mcp", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 405
    batch = client.post("/agent/mcp", headers={"Authorization": f"Bearer {TOKEN}"}, json=[
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 7, "method": "ping"}])
    assert batch.get_json() == [{"jsonrpc": "2.0", "id": 7, "result": {}}]
    assert rpc(client, "no/such").get_json()["error"]["code"] == mcp.METHOD_NOT_FOUND


def test_tools_are_listed_with_schemas(reports, client):
    tools = {t["name"]: t for t in rpc(client, "tools/list").get_json()["result"]["tools"]}
    assert set(tools) == {"list_reports", "triage_brief", "student_detail", "save_triage", "grade_proposal"}
    assert tools["save_triage"]["inputSchema"]["required"] == ["triage"]
    assert tools["save_triage"]["annotations"]["readOnlyHint"] is False
    assert tools["triage_brief"]["annotations"]["readOnlyHint"] is True


# --- Tools -------------------------------------------------------------------

def test_list_reports_newest_first(reports, client):
    result = call(client, "list_reports")
    names = [r["report"] for r in result["structuredContent"]["reports"]]
    assert names == [NEW, OLD]
    only = call(client, "list_reports", assignment="lab 0")["structuredContent"]["reports"]
    assert [r["report"] for r in only] == [OLD]


def test_triage_brief_is_coded(reports, client, sample_report):
    result = call(client, "triage_brief")                    # 'latest' by default
    text = result["content"][0]["text"]
    assert text.startswith("# Triage brief: Lab 1") and NEW in text
    for s in sample_report["students"].values():
        assert s["name"] not in text and s["email"] not in text
    assert result["structuredContent"]["students"][0]["tier"] == "act"
    assert call(client, "triage_brief", report=OLD[:-5])["content"][0]["text"].startswith("# Triage brief: Lab 0")


def test_bad_input_comes_back_as_a_tool_error(reports, client):
    for args in ({"report": "nope"}, {"report": "../app.py"}):
        result = call(client, "triage_brief", **args)
        assert result["isError"] and "list_reports" in result["content"][0]["text"]
    assert call(client, "student_detail", student="S77")["isError"]
    assert call(client, "student_detail")["isError"]                       # student is required
    assert call(client, "triage_brief", surprise=1)["isError"]
    assert rpc(client, "tools/call", {"name": "delete_everything"}).get_json()["error"]["code"] == mcp.INVALID_PARAMS


def test_save_triage_stores_notes_and_proposes_held_grades(reports, client):
    result = call(client, "save_triage", triage="## Act now\n- S04 and S02 look copied.", hold="S04, S99")
    assert not result["isError"]
    assert result["structuredContent"]["hold"] == ["S04"]
    assert "/report/" + NEW in result["structuredContent"]["url"]

    path = str(reports / NEW)
    stored = triage.load_triage(path)
    assert stored["state"] == "done" and stored["hold"] == ["S04"] and stored["source"] == "agent"
    proposal = grades.load(path)
    held = {r["uid"]: r for r in proposal["rows"] if r["held"]}
    assert set(held) == {"b"}
    assert held["b"]["reason"].startswith("held by the agent: similar to S02")
    assert proposal["created_by"] == "agent" and proposal["status"] == grades.PENDING

    status = call(client, "grade_proposal")
    assert "S04: 50.0% -> 5.0 (held)" in status["content"][0]["text"]


def test_save_triage_never_overwrites_saved_grades(reports, client):
    path = str(reports / NEW)
    proposal = grades.propose(json.loads((reports / NEW).read_text()))
    proposal["status"] = grades.APPLIED
    grades.save(path, proposal)
    text = call(client, "save_triage", triage="notes", hold="S02")["content"][0]["text"]
    assert "already saved" in text
    assert grades.load(path)["status"] == grades.APPLIED


def test_json_api_mirrors_the_tools(reports, client):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    listed = client.get("/agent/tools", headers=headers).get_json()["tools"]
    assert len(listed) == len(agent_tools.TOOLS)
    ok = client.post("/agent/tools/triage_brief", json={"report": "latest"}, headers=headers).get_json()
    assert ok["text"].startswith("# Triage brief")
    bad = client.post("/agent/tools/student_detail", json={"student": "S99"}, headers=headers)
    assert bad.status_code == 400
    assert client.post("/agent/tools/nope", json={}, headers=headers).status_code == 404


# --- Report page: follow-up and approving draft grades -----------------------

class GradingClassroom:
    saved = []

    def set_draft_grade(self, course_id, work_id, sub_id, grade):
        GradingClassroom.saved.append((sub_id, grade))


def _sign_in(client, write=True):
    with client.session_transaction() as s:
        s["credentials"] = {"token": "x", "scopes": [COURSEWORK_WRITE] if write else []}
        s["csrf"] = "c"


def test_report_page_shows_follow_up_with_names(reports, client):
    _sign_in(client)
    call(client, "save_triage", triage="S04 should explain the overlap with S02.", hold="S04")
    html = client.get("/report/" + NEW).data.decode()
    assert "Follow-up" in html and "Needs action" in html
    assert "Trần Bình (S04) should explain the overlap with Nguyễn Văn An (S02)." in html
    assert "No concerns: Phạm Dũng" in html
    assert "Save ticked draft grades to Classroom" in html


def test_teacher_approves_and_saves_draft_grades(reports, client, monkeypatch):
    monkeypatch.setattr(webapp, "_credentials", lambda: object())
    monkeypatch.setattr(webapp, "Classroom", lambda creds: GradingClassroom())
    GradingClassroom.saved = []
    _sign_in(client)

    assert client.post(f"/report/{NEW}/grades/propose", data={"max_points": "10"}).status_code == 400  # no csrf
    assert client.post(f"/report/{NEW}/grades/propose",
                       data={"max_points": "20", "csrf": "c"}).status_code == 302
    assert grades.load(str(reports / NEW))["rows"][0]["grade"] == 20

    resp = client.post(f"/report/{NEW}/grades/apply", data={"csrf": "c", "approve": ["a", "d"]})
    assert resp.status_code == 302
    assert sorted(GradingClassroom.saved) == [("sa", 20.0), ("sd", 20.0)]
    saved = grades.load(str(reports / NEW))
    assert saved["status"] == grades.APPLIED
    assert {r["uid"]: r["result"] for r in saved["rows"]} == {"a": "saved", "b": "skipped", "d": "saved"}

    # Once saved, a new proposal from the agent or the page may not replace it.
    assert client.post(f"/report/{NEW}/grades/propose",
                       data={"max_points": "10", "csrf": "c"}).status_code == 400
    csv_text = client.get(f"/report/{NEW}/grades.csv").data.decode("utf-8-sig")
    assert csv_text.splitlines()[0].startswith("Student,Email,Score %,Proposed grade")


def test_saving_needs_grade_access(reports, client, monkeypatch):
    monkeypatch.setattr(webapp, "_credentials", lambda: object())
    monkeypatch.setattr(webapp, "Classroom", lambda creds: GradingClassroom())
    _sign_in(client, write=False)
    grades.save(str(reports / NEW), grades.propose(json.loads((reports / NEW).read_text())))
    assert client.post(f"/report/{NEW}/grades/apply", data={"csrf": "c", "approve": ["a"]}).status_code == 403
    html = client.get("/report/" + NEW).data.decode()
    assert "GRADE_WRITE=1" in html and "Save ticked draft grades" not in html


def test_assignments_made_in_classroom_point_to_the_csv(reports, client):
    report = json.loads((reports / NEW).read_text())
    report["work"]["associated_with_developer"] = False
    (reports / NEW).write_text(json.dumps(report))
    grades.save(str(reports / NEW), grades.propose(report))
    _sign_in(client)
    html = client.get("/report/" + NEW).data.decode()
    assert "same app created" in html and "Save ticked draft grades" not in html


# --- Triage through Dify -----------------------------------------------------

class FakeDify(BaseHTTPRequestHandler):
    mode = "save"            # save | silent | unauthorized
    received = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeDify.received.append((self.headers.get("Authorization"), body))
        if FakeDify.mode == "unauthorized":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"code": "unauthorized"}')
            return
        if FakeDify.mode == "save":            # what the workflow does through MCP
            agent_tools.call("save_triage", {"report": body["inputs"]["report"],
                                             "triage": "S04: ask about the overlap.", "hold": "S04"})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [{"event": "workflow_started", "data": {}}, "ping",
                  {"event": "workflow_finished", "data": {"status": "succeeded", "outputs": {}}}]
        for e in events:
            line = "event: ping\n\n" if e == "ping" else f"data: {json.dumps(e)}\n\n"
            self.wfile.write(line.encode())


@pytest.fixture
def fake_dify(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeDify)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(dify, "DIFY_API_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(dify, "DIFY_TRIAGE_KEY", "app-test")
    FakeDify.received.clear()
    yield FakeDify
    server.shutdown()


def test_run_triage_records_the_outcome(reports, fake_dify):
    path = str(reports / NEW)
    fake_dify.mode = "save"
    assert dify.run_triage(path)["state"] == "done"
    auth, body = fake_dify.received[-1]
    assert auth == "Bearer app-test"
    assert body["inputs"]["report"] == NEW and body["response_mode"] == "streaming"

    fake_dify.mode = "silent"
    failed = dify.run_triage(path)
    assert failed["state"] == "failed" and "without calling save_triage" in failed["error"]

    fake_dify.mode = "unauthorized"
    assert "Dify returned 401" in dify.run_triage(path)["error"]


def test_dify_unreachable(reports, monkeypatch):
    monkeypatch.setattr(dify, "DIFY_API_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(dify, "DIFY_TRIAGE_KEY", "app-test")
    result = dify.run_triage(str(reports / NEW))
    assert result["state"] == "failed" and "Cannot reach Dify" in result["error"]


def test_triage_button_runs_in_the_background(reports, client, fake_dify):
    fake_dify.mode = "save"
    _sign_in(client)
    assert client.post(f"/report/{NEW}/triage", data={"csrf": "c"}).status_code == 302
    deadline = time.time() + 10
    while time.time() < deadline and (triage.load_triage(str(reports / NEW)) or {}).get("state") != "done":
        time.sleep(0.05)
    assert triage.load_triage(str(reports / NEW))["state"] == "done"


def test_triage_runs_after_each_check(fake_ollama, fake_dify, monkeypatch, tmp_path):
    """The real browser flow: run a check, and the agent triages the new report by itself."""
    from checker import runner
    from test_end_to_end import FakeWebClassroom, run_check

    monkeypatch.setattr(runner, "CODE_RUNNER", "local")
    monkeypatch.setattr(checks, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(webapp, "_credentials", lambda: object())
    monkeypatch.setattr(webapp, "Classroom", lambda creds: FakeWebClassroom())
    fake_dify.mode = "save"
    client = webapp.app.test_client()
    with client.session_transaction() as s:
        s["credentials"] = {"token": "x"}

    status, _ = run_check(client, {"duplicates": "on", "numbers": "on", "use_llm": "on",
                                   "answer_key": "1: 42\n2: 12.5"})
    path = str(tmp_path / status["report_file"])
    deadline = time.time() + 10
    while time.time() < deadline and (triage.load_triage(path) or {}).get("state") != "done":
        time.sleep(0.05)
    assert triage.load_triage(path)["state"] == "done"
    assert fake_dify.received[-1][1]["inputs"]["report"] == status["report_file"]
    assert "Follow-up" in client.get("/report/" + status["report_file"]).data.decode()


def test_cli_triage(reports, fake_dify, capsys):
    from checker import cli
    assert cli.main(["triage"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Triage brief: Lab 1") and "Trần Bình (S04)" in out

    fake_dify.mode = "save"
    assert cli.main(["triage", "--report", NEW, "--agent"]) == 0
    out = capsys.readouterr().out
    assert "-- Agent triage --" in out and "Trần Bình (S04): ask about the overlap." in out
    assert "/report/" + NEW in out
    assert cli.main(["triage", "--report", "missing"]) == 1
