"""Google Classroom assignment checker - Flask web app."""
from __future__ import annotations

import csv
import hmac
import io
import json
import os
import re
import secrets
import threading
import uuid
from collections import Counter
from functools import wraps

from flask import (Flask, Response, abort, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.errors import HttpError

from checker import agent_tools, answerkey, checks, dify, grades, llm, mcp, results, runner, triage
from checker import status as st
from checker.classroom import GRADE_WRITE, SCOPES, Classroom, can_write_grades

# Google may return previously granted scopes too; don't treat that as an error.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
CLIENT_SECRETS = os.environ.get("GOOGLE_CLIENT_SECRETS", "client_secret.json")


# --- Auth -------------------------------------------------------------------

def _flow(state: str | None = None) -> Flow:
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS, scopes=SCOPES, state=state)
    flow.redirect_uri = url_for("oauth2callback", _external=True)
    return flow


def _credentials() -> Credentials | None:
    data = session.get("credentials")
    if not data:
        return None
    creds = Credentials(**data)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _store(creds)
    return creds


def _store(creds: Credentials) -> None:
    session["credentials"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        # What Google actually granted: on its consent screen the teacher may untick
        # the grade permission, and the Save button must then not appear.
        "scopes": list(getattr(creds, "granted_scopes", None) or creds.scopes or SCOPES),
    }


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        try:
            creds = _credentials()
        except RefreshError:
            creds = None
        if creds is None:
            session.clear()
            return redirect(url_for("index"))
        try:
            return view(Classroom(creds), *args, **kwargs)
        except RefreshError:
            session.clear()
            return redirect(url_for("index"))
        except HttpError as e:
            return render_template("error.html", error=e), e.status_code
    return wrapper


@app.route("/login")
def login():
    if not os.path.exists(CLIENT_SECRETS):
        return render_template("setup.html", path=CLIENT_SECRETS), 500
    flow = _flow()
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent")
    session["state"] = state
    session["code_verifier"] = flow.code_verifier
    return redirect(auth_url)


@app.route("/oauth2callback")
def oauth2callback():
    flow = _flow(state=session.pop("state", None))
    flow.code_verifier = session.pop("code_verifier", None)
    flow.fetch_token(authorization_response=request.url)
    _store(flow.credentials)
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# --- Data helpers -----------------------------------------------------------

def _work_summary(work: dict, subs: list[dict]) -> dict:
    due = st.due_datetime(work)
    counts = Counter(st.submission_status(s, due) for s in subs)
    return {
        "work": work,
        "due": due,
        "counts": counts,
        "done": sum(counts[k] for k in st.DONE),
        "total": len(subs),
    }


def _course_matrix(cr: Classroom, course_id: str):
    students = cr.students(course_id)
    works = cr.course_work(course_id)
    works.sort(key=lambda w: w.get("creationTime", ""), reverse=True)
    subs = cr.submissions(course_id)

    by_work: dict[str, list[dict]] = {}
    cell: dict[tuple[str, str], dict] = {}
    for s in subs:
        by_work.setdefault(s["courseWorkId"], []).append(s)
        cell[(s["userId"], s["courseWorkId"])] = s

    summaries = [_work_summary(w, by_work.get(w["id"], [])) for w in works]
    rows = []
    for stu in students:
        uid = stu["userId"]
        cells, missing = [], 0
        for summ in summaries:
            sub = cell.get((uid, summ["work"]["id"]))
            status = st.submission_status(sub, summ["due"]) if sub else None
            missing += status == st.MISSING
            cells.append({"sub": sub, "status": status, "grade": st.grade_of(sub) if sub else None})
        rows.append({"student": stu, "cells": cells, "missing": missing})
    return students, summaries, rows


# --- Views ------------------------------------------------------------------

@app.route("/")
def index():
    if "credentials" not in session:
        return render_template("landing.html")
    return courses()


@login_required
def courses(cr: Classroom):
    return render_template("courses.html", courses=cr.courses())


@app.route("/course/<course_id>")
@login_required
def course(cr: Classroom, course_id: str):
    students, summaries, rows = _course_matrix(cr, course_id)
    if request.args.get("filter") == "missing":
        rows = [r for r in rows if r["missing"]]
    return render_template("course.html", course=cr.course(course_id), summaries=summaries,
                           rows=rows, student_count=len(students), labels=st.LABELS)


@app.route("/course/<course_id>/export.csv")
@login_required
def export_csv(cr: Classroom, course_id: str):
    _, summaries, rows = _course_matrix(cr, course_id)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Student", "Email", "Missing"]
                    + [s["work"]["title"] for s in summaries])
    for r in rows:
        prof = r["student"]["profile"]
        values = []
        for c in r["cells"]:
            if c["status"] is None:
                values.append("")
            elif c["grade"] is not None:
                values.append(f'{st.LABELS[c["status"]]} ({c["grade"]:g})')
            else:
                values.append(st.LABELS[c["status"]])
        writer.writerow([prof["name"].get("fullName", ""), prof.get("emailAddress", ""),
                         r["missing"]] + values)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=course-{course_id}.csv"})


def _assignment_rows(cr: Classroom, course_id: str, work: dict):
    students = {s["userId"]: s for s in cr.students(course_id)}
    subs = cr.submissions(course_id, work["id"])
    due = st.due_datetime(work)
    rows = []
    for s in subs:
        stu = students.get(s["userId"])
        if stu is None:  # student has left the course
            continue
        rows.append({
            "student": stu,
            "sub": s,
            "status": st.submission_status(s, due),
            "turned_in_at": st.turned_in_at(s),
            "grade": st.grade_of(s),
            "attachments": st.attachments_of(s),
            "answer": st.answer_text(s),
        })
    order = [st.MISSING, st.ASSIGNED, st.LATE, st.TURNED_IN, st.RETURNED, st.GRADED]
    rows.sort(key=lambda r: (order.index(r["status"]),
                             r["student"]["profile"]["name"].get("fullName", "").lower()))
    return rows, due


@app.route("/course/<course_id>/work/<work_id>")
@login_required
def assignment(cr: Classroom, course_id: str, work_id: str):
    work = cr.one_course_work(course_id, work_id)
    rows, due = _assignment_rows(cr, course_id, work)
    status_filter = request.args.get("status")
    counts = Counter(r["status"] for r in rows)
    if status_filter:
        rows = [r for r in rows if r["status"] == status_filter]
    return render_template("assignment.html", course=cr.course(course_id), work=work, due=due,
                           rows=rows, counts=counts, labels=st.LABELS, status_filter=status_filter)


@app.route("/course/<course_id>/work/<work_id>/submission/<sub_id>")
@login_required
def submission(cr: Classroom, course_id: str, work_id: str, sub_id: str):
    work = cr.one_course_work(course_id, work_id)
    sub = cr.submission(course_id, work_id, sub_id)
    student = cr.student(course_id, sub["userId"])
    files = []
    for att in st.attachments_of(sub):
        text, note = None, ""
        if att["file_id"]:
            info = cr.file_content(att["file_id"])
            text, note = info["text"], info["note"]
            if text and (info["name"].lower().endswith(".ipynb") or "colaboratory" in info["mime"]):
                try:
                    text = runner.notebook_to_script(text)
                except ValueError:
                    pass
        files.append({**att, "text": text, "note": note,
                      "words": len(text.split()) if text else 0})
    return render_template("submission.html", course=cr.course(course_id), work=work, sub=sub,
                           student=student, files=files, answer=st.answer_text(sub),
                           status=st.submission_status(sub, st.due_datetime(work)),
                           turned_in_at=st.turned_in_at(sub), grade=st.grade_of(sub),
                           labels=st.LABELS)


@app.route("/course/<course_id>/work/<work_id>/check", methods=["GET", "POST"])
@login_required
def check(cr: Classroom, course_id: str, work_id: str):
    work = cr.one_course_work(course_id, work_id)
    course_info = cr.course(course_id)

    def form_page(upload_error: str | None = None, code: int = 200):
        return render_template("check.html", course=course_info, work=work,
                               ollama=llm.status(), runner=runner.runner_available(),
                               all_checks=checks.ALL_CHECKS, models=llm.installed(),
                               upload_error=upload_error), code

    if request.method == "GET":
        return form_page()
    form = request.form

    model_answer = form.get("model_answer", "")
    upload = request.files.get("model_answer_file")
    if upload and upload.filename:
        try:
            model_answer = answerkey.read_upload(upload.filename, upload.read())
        except ValueError as e:
            return form_page(str(e), 400)

    opts = checks.Options(
        checks=tuple(c for c in checks.ALL_CHECKS if form.get(c)),
        use_llm=bool(form.get("use_llm")),
        model=form.get("model", ""),
        embed_model=form.get("embed_model", ""),
        answer_key=form.get("answer_key", ""),
        model_answer=model_answer,
        tests=form.get("tests", ""),
        dup_threshold=float(form.get("threshold") or 0.3),
        include_unsubmitted=bool(form.get("include_unsubmitted")),
    )
    # A full check takes minutes, so it runs in a thread and the browser watches a
    # progress bar instead of holding one very long request open.
    job_id = start_job(_credentials(), course_id, work_id, opts)
    return redirect(url_for("job_page", job_id=job_id))


# --- Background check jobs --------------------------------------------------

# ponytail: in-memory, and the app runs one gunicorn worker (see the Dockerfile), so
# every request sees the same dict. Jobs are lost on restart; reports on disk are not.
JOBS: dict[str, dict] = {}
MAX_JOBS = 50


def start_job(creds, course_id: str, work_id: str, opts: checks.Options) -> str:
    job_id = uuid.uuid4().hex[:12]
    state = {"id": job_id, "state": "running", "phase": "Starting", "message": "",
             "done": 0, "total": 0, "report_file": None, "error": None,
             "course_id": course_id, "work_id": work_id,
             "checks": list(opts.checks)}
    JOBS[job_id] = state
    for old in list(JOBS)[:-MAX_JOBS]:          # keep the recent ones, drop the rest
        JOBS.pop(old, None)

    def worker():
        try:
            progress = checks.Progress(
                lambda phase, message, done, total: state.update(
                    phase=phase, message=message, done=done, total=total))
            report = checks.run(Classroom(creds), course_id, work_id, opts, progress=progress)
            path = checks.save(report)
            results.save_csv(report, path[:-len(".json")] + ".csv")
            state["report_file"] = os.path.basename(path)
            state["state"] = "done"
        except Exception as e:                  # noqa: BLE001 - a thread must not die silently
            state["state"] = "error"
            state["error"] = f"{type(e).__name__}: {e}"
            return
        # The report is ready and the browser moves on to it; the agent's triage
        # follows in the same thread and the report page shows it when it lands.
        if dify.configured():
            state["triage"] = dify.run_triage(path).get("state")

    threading.Thread(target=worker, daemon=True, name=f"check-{job_id}").start()
    return job_id


def _job_or_404(job_id: str) -> dict:
    if "credentials" not in session:
        abort(401)
    state = JOBS.get(job_id)
    if state is None:
        abort(404)
    return state


@app.route("/job/<job_id>")
def job_page(job_id: str):
    return render_template("job.html", job=_job_or_404(job_id))


@app.route("/job/<job_id>/status")
def job_status(job_id: str):
    return _job_or_404(job_id)


# --- Saved reports ----------------------------------------------------------

SAFE_REPORT_NAME = re.compile(r"^[\w.-]+\.json$")


def _load_report(filename: str) -> dict:
    if "credentials" not in session:
        abort(401)
    if not SAFE_REPORT_NAME.match(filename):
        abort(404)
    path = os.path.join(os.path.abspath(checks.REPORTS_DIR), filename)
    if not os.path.isfile(path):
        abort(404)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _report_path(filename: str) -> str:
    return os.path.join(os.path.abspath(checks.REPORTS_DIR), filename)


def _csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(24)
    return session["csrf"]


def _same_secret(sent: str, expected: str) -> bool:
    # Bytes, because compare_digest rejects non-ASCII str with an error.
    return bool(sent) and bool(expected) and hmac.compare_digest(sent.encode(), expected.encode())


def _check_csrf() -> None:
    if not _same_secret(request.form.get("csrf", ""), session.get("csrf", "")):
        abort(400, "This form has expired. Reload the page and try again.")


def _session_can_write() -> bool:
    return can_write_grades((session.get("credentials") or {}).get("scopes"))


def _follow_up(report: dict) -> list[dict]:
    """The deterministic triage, with names put back for the teacher."""
    b = triage.brief(report)
    uid_of = {c: u for u, c in triage.codes(report).items()}
    rows = []
    for p in b["students"]:
        uid = uid_of[p["code"]]
        rows.append({**p, "uid": uid, "name": report["students"][uid]["name"],
                     "signals": [{**s, "text": triage.with_names(s["text"], report)}
                                 for s in p["signals"]]})
    return rows


@app.route("/report/<filename>")
def report_page(filename: str):
    report = _load_report(filename)
    path = _report_path(filename)
    agent = triage.current_state(triage.load_triage(path), dify.DIFY_TIMEOUT)
    if agent and agent.get("text"):
        agent = {**agent, "text": triage.with_names(agent["text"], report)}
    return render_template("report.html", course=report.get("course") or {"id": report["course_id"]},
                           work=report["work"], report=report,
                           report_file=filename, labels=st.LABELS,
                           follow_up=_follow_up(report), codes=triage.codes(report),
                           tier_labels=triage.TIER_LABELS, agent=agent,
                           proposal=grades.load(path), grade_write=GRADE_WRITE,
                           can_write=_session_can_write(), dify_on=dify.configured(),
                           triage_names=lambda text: triage.with_names(text, report),
                           csrf=_csrf_token())


@app.route("/report/<filename>/triage.json")
def report_triage_state(filename: str):
    _load_report(filename)
    agent = triage.current_state(triage.load_triage(_report_path(filename)), dify.DIFY_TIMEOUT) or {}
    return jsonify({"state": agent.get("state"), "updated_at": agent.get("updated_at")})


@app.route("/report/<filename>/triage", methods=["POST"])
def report_triage(filename: str):
    _load_report(filename)
    _check_csrf()
    if not dify.configured():
        abort(400, "Set DIFY_TRIAGE_KEY to use the triage agent.")
    path = _report_path(filename)
    triage.write_triage(path, state="running", error=None)
    threading.Thread(target=dify.run_triage, args=(path,), daemon=True,
                     name=f"triage-{filename[:20]}").start()
    return redirect(url_for("report_page", filename=filename) + "#follow-up")


@app.route("/report/<filename>/grades/propose", methods=["POST"])
def grades_propose(filename: str):
    report = _load_report(filename)
    _check_csrf()
    path = _report_path(filename)
    try:
        max_points = float(request.form.get("max_points") or 0) or None
    except ValueError:
        max_points = None
    # Keep any holds the agent asked for when the teacher re-proposes.
    held = {r["uid"]: r["reason"] for r in (grades.load(path) or {}).get("rows", []) if r["held"]}
    try:
        grades.replace(path, grades.propose(report, max_points, hold=held))
    except ValueError as e:                     # no usable points, or already saved
        abort(400, str(e))
    return redirect(url_for("report_page", filename=filename) + "#draft-grades")


@app.route("/report/<filename>/grades/apply", methods=["POST"])
@login_required
def grades_apply(cr: Classroom, filename: str):
    _load_report(filename)
    _check_csrf()
    if not _session_can_write():
        abort(403, "Sign in with grade access (GRADE_WRITE=1) to save draft grades.")
    try:
        grades.apply_reviewed(cr, _report_path(filename), request.form.get("revision", ""),
                              set(request.form.getlist("approve")))
    except grades.StaleProposal as e:
        abort(409, str(e))
    return redirect(url_for("report_page", filename=filename) + "#draft-grades")


@app.route("/report/<filename>/grades.csv")
def grades_csv(filename: str):
    _load_report(filename)
    proposal = grades.load(_report_path(filename))
    if not proposal:
        abort(404)
    name = filename[:-len(".json")] + "_draft-grades.csv"
    return Response("﻿" + grades.to_csv(proposal), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# --- Agent tools (MCP) --------------------------------------------------------

# Off unless AGENT_TOKEN is set. Anything holding the token can read every saved
# report (students as codes only) and propose draft grades, so keep it secret and
# keep this port on your own machine or network.
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "")


def _agent_allowed() -> bool:
    if not AGENT_TOKEN:
        abort(404)
    # Header only: a token in the URL would end up in access logs.
    sent = request.headers.get("Authorization", "")
    sent = sent[7:].strip() if sent.lower().startswith("bearer ") else ""
    return _same_secret(sent, AGENT_TOKEN)


def _unauthorized():
    resp = jsonify(mcp.error_response(None, -32001, "Unauthorized"))
    resp.status_code = 401
    resp.headers["WWW-Authenticate"] = 'Bearer realm="ggclassroom-checker"'
    return resp


@app.route("/agent/mcp", methods=["GET", "POST", "DELETE"])
def agent_mcp():
    if not _agent_allowed():
        return _unauthorized()
    if request.method != "POST":
        # No server-to-client stream and no sessions to end: allowed by the spec.
        return Response(status=405, headers={"Allow": "POST"})
    try:
        body = json.loads(request.get_data(as_text=True) or "null")
    except ValueError:
        return jsonify(mcp.error_response(None, mcp.PARSE_ERROR, "Parse error")), 400
    try:
        reply = mcp.handle_body(body)
    except Exception as e:                       # noqa: BLE001 - answer, don't 500
        msg_id = body.get("id") if isinstance(body, dict) else None
        reply = mcp.error_response(msg_id, -32603, f"Internal error: {type(e).__name__}: {e}")
    if reply is None:
        return Response(status=202)
    # No Mcp-Session-Id: without one, clients do not open a GET stream or send DELETE.
    return jsonify(reply)


@app.route("/agent/tools")
def agent_tool_list():
    if not _agent_allowed():
        return _unauthorized()
    return jsonify({"tools": mcp.tool_list()})


@app.route("/agent/tools/<name>", methods=["POST"])
def agent_tool_call(name: str):
    """The same tools as plain JSON, for curl, n8n or a Dify HTTP node."""
    if not _agent_allowed():
        return _unauthorized()
    if name not in agent_tools.TOOLS:
        abort(404)
    try:
        text, data = agent_tools.call(name, request.get_json(silent=True))
    except agent_tools.ToolError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"text": text, "data": data})


@app.route("/report/<filename>/csv")
def report_csv(filename: str):
    report = _load_report(filename)
    name = filename[:-len(".json")] + ".csv"
    # The BOM makes Excel on Windows read the Vietnamese names as UTF-8.
    return Response("\ufeff" + results.to_csv(report),
                    mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.route("/reports/<path:filename>")
def download_report(filename: str):
    if "credentials" not in session:
        abort(401)
    return send_from_directory(os.path.abspath(checks.REPORTS_DIR), filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "localhost"), port=int(os.environ.get("PORT", 8080)),
            debug=os.environ.get("FLASK_DEBUG", "1") == "1")
