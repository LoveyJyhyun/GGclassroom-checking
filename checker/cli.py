"""Command-line interface.

    python -m checker.cli courses
    python -m checker.cli works   --course COURSE_ID
    python -m checker.cli check   --course COURSE_ID --work WORK_ID \\
        --checks duplicates,numbers,ai,code --key key.txt --tests tests.txt
"""
from __future__ import annotations

import argparse
import os
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from . import checks, llm, runner
from . import status as st
from .classroom import SCOPES, Classroom

CLIENT_SECRETS = os.environ.get("GOOGLE_DESKTOP_CLIENT_SECRETS",
                                os.environ.get("GOOGLE_CLIENT_SECRETS", "client_secret.json"))
TOKEN_FILE = os.environ.get("GOOGLE_TOKEN_FILE", "token.json")


def credentials() -> Credentials:
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
        # In Docker: set OAUTH_PORT (published to the host) and OAUTH_NO_BROWSER=1,
        # then open the printed URL in your browser.
        in_container = os.environ.get("OAUTH_NO_BROWSER") == "1"
        creds = flow.run_local_server(
            port=int(os.environ.get("OAUTH_PORT", "0")),
            open_browser=not in_container,
            bind_addr="0.0.0.0" if in_container else None)
    os.makedirs(os.path.dirname(os.path.abspath(TOKEN_FILE)), exist_ok=True)
    with open(TOKEN_FILE, "w") as fh:
        fh.write(creds.to_json())
    os.chmod(TOKEN_FILE, 0o600)
    return creds


def read_arg_file(path: str | None) -> str:
    if not path:
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def print_summary(report: dict) -> None:
    names = {uid: s["name"] for uid, s in report["students"].items()}
    print(f"\n== {report['work']['title']} ({len(names)} submissions) ==")

    if report["duplicates"] is not None:
        print("\n-- Possible copying --")
        if not report["duplicates"]:
            print("  none found")
        for p in report["duplicates"]:
            parts = [f"{k} {p[k]:.0%}" for k in ("lexical", "semantic", "code") if p[k] is not None]
            verdict = ""
            if p["llm"]:
                verdict = f" | LLM: {'copied' if p['llm']['copied'] else 'not copied'} ({p['llm']['confidence']}%)"
            print(f"  {names[p['a']]}  <->  {names[p['b']]}: {', '.join(parts)}{verdict}")

    if report["numbers"]:
        print("\n-- Numeric answers --")
        for uid, r in report["numbers"]["results"].items():
            wrong = [x["question"] for x in r["results"] if not x["ok"]]
            print(f"  {names[uid]}: {r['correct']}/{r['total']}"
                  + (f"  wrong/missing: {', '.join(wrong)}" if wrong else ""))

    if report["ai"]:
        print("\n-- AI-writing indicators (not proof) --")
        for uid, r in report["ai"].items():
            h = r["heuristic"]["score"]
            l = r["llm"]["likelihood"] if r["llm"] else None
            print(f"  {names[uid]}: heuristic {'n/a' if h is None else f'{h}%'}"
                  f", LLM {'n/a' if l is None else f'{l}%'}")

    if report["code"]:
        print("\n-- Code --")
        for uid, r in report["code"]["results"].items():
            line = f"  {names[uid]}: {r['passed']}/{r['total']} tests passed"
            if r.get("llm"):
                line += f" | LLM: {r['llm']['summary']}"
            if r.get("error"):
                line += f" | {r['error']}"
            print(line)

    for e in report["errors"]:
        print(f"\n! {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m checker.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("courses", help="list your active courses")
    p_works = sub.add_parser("works", help="list assignments in a course")
    p_works.add_argument("--course", required=True)
    sub.add_parser("status", help="check Ollama and the code runner")
    p_check = sub.add_parser("check", help="run checks on one assignment")
    p_check.add_argument("--course", required=True)
    p_check.add_argument("--work", required=True)
    p_check.add_argument("--checks", default=",".join(checks.ALL_CHECKS),
                         help="comma list of: " + ", ".join(checks.ALL_CHECKS))
    p_check.add_argument("--key", help="answer key file for the numbers check")
    p_check.add_argument("--tests", help="test cases file for the code check")
    p_check.add_argument("--no-llm", action="store_true", help="heuristics only, no Ollama calls")
    p_check.add_argument("--threshold", type=float, default=0.3, help="copy-detection threshold (0-1)")
    p_check.add_argument("--include-unsubmitted", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "status":
        s = llm.status()
        print(f"Ollama: {'OK' if s['ok'] else 'NOT REACHABLE'} at {s['url']}")
        if s["ok"]:
            print(f"  chat model  {s['model']}: {'installed' if s['model_ok'] else 'missing -> ollama pull ' + s['model']}")
            print(f"  embed model {s['embed_model']}: {'installed' if s['embed_ok'] else 'missing -> ollama pull ' + s['embed_model']}")
        ok, note = runner.runner_available()
        print(f"Code runner: {note}" if ok else f"Code runner: unavailable - {note}")
        return 0

    cr = Classroom(credentials())
    if args.cmd == "courses":
        for c in cr.courses():
            print(f"{c['id']}\t{c['name']}\t{c.get('section', '')}")
    elif args.cmd == "works":
        for w in cr.course_work(args.course):
            due = st.due_datetime(w)
            print(f"{w['id']}\t{w['title']}\t{due.isoformat() if due else 'no due date'}")
    else:
        selected = tuple(c.strip() for c in args.checks.split(",") if c.strip())
        unknown = set(selected) - set(checks.ALL_CHECKS)
        if unknown:
            parser.error(f"unknown checks: {', '.join(sorted(unknown))}")
        opts = checks.Options(checks=selected, use_llm=not args.no_llm,
                              answer_key=read_arg_file(args.key), tests=read_arg_file(args.tests),
                              dup_threshold=args.threshold,
                              include_unsubmitted=args.include_unsubmitted)
        report = checks.run(cr, args.course, args.work, opts,
                            progress=lambda m: print(f"... {m}", file=sys.stderr))
        print_summary(report)
        print(f"\nFull report: {checks.save(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
