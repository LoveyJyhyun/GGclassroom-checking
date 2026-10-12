"""Google Classroom assignment checker - Flask web app."""
from __future__ import annotations

import csv
import io
import os
from collections import Counter
from functools import wraps

from flask import (Flask, Response, abort, redirect, render_template, request,
                   send_from_directory, session, url_for)
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.errors import HttpError

from checker import checks, llm, runner
from checker import status as st
from checker.classroom import SCOPES, Classroom

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
        "scopes": list(creds.scopes or SCOPES),
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
    if request.method == "GET":
        return render_template("check.html", course=course_info, work=work,
                               ollama=llm.status(), runner=runner.runner_available(),
                               all_checks=checks.ALL_CHECKS)
    form = request.form
    opts = checks.Options(
        checks=tuple(c for c in checks.ALL_CHECKS if form.get(c)),
        use_llm=bool(form.get("use_llm")),
        answer_key=form.get("answer_key", ""),
        tests=form.get("tests", ""),
        dup_threshold=float(form.get("threshold") or 0.3),
        include_unsubmitted=bool(form.get("include_unsubmitted")),
    )
    report = checks.run(cr, course_id, work_id, opts)
    path = checks.save(report)
    return render_template("report.html", course=course_info, work=work, report=report,
                           report_file=os.path.basename(path), labels=st.LABELS)


@app.route("/reports/<path:filename>")
def download_report(filename: str):
    if "credentials" not in session:
        abort(401)
    return send_from_directory(os.path.abspath(checks.REPORTS_DIR), filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "localhost"), port=int(os.environ.get("PORT", 8080)),
            debug=os.environ.get("FLASK_DEBUG", "1") == "1")
