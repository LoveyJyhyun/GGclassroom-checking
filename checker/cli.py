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

from . import agent_tools, answerkey, checks, dify, llm, results, runner, triage
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


class Bar:
    """One-line progress bar on stderr, so stdout stays pipeable."""

    WIDTH = 28

    def __init__(self, stream=sys.stderr):
        self.stream = stream
        self.live = stream.isatty()
        self.width = 0

    def __call__(self, phase: str, message: str, done: int, total: int) -> None:
        if total:
            filled = round(self.WIDTH * done / total)
            bar = "#" * filled + "-" * (self.WIDTH - filled)
            head = f"[{bar}] {done:>3}/{total}"
        else:
            head = f"[{'?' * self.WIDTH}]  ...  "
        line = f"{head}  {phase}{': ' + message if message else ''}"[:110]
        if self.live:
            # Pad to the previous width instead of an ANSI erase: the old Windows
            # console does not process escape codes and would print them literally.
            self.stream.write("\r" + line.ljust(self.width))
            self.width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def close(self) -> None:
        if self.live:
            self.stream.write("\r" + " " * self.width + "\r")
        self.stream.flush()


def print_scores(report: dict) -> None:
    marks = {uid: s for uid, s in (report.get("scores") or {}).items() if s}
    if not marks:
        return
    names = {uid: s["name"] for uid, s in report["students"].items()}
    print("\n-- Final score (from the answer sheet) --")
    for uid, sc in sorted(marks.items(), key=lambda kv: -kv[1]["percent"]):
        wrong = [q for q, v in sorted(sc["per_question"].items()) if v < 0.999]
        print(f"  {names[uid]}: {sc['percent']:5.1f}%  ({sc['earned']}/{sc['questions']})"
              + (f"  lost marks on: {', '.join(wrong)}" if wrong else ""))


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
    # Student names and answer keys are often Vietnamese; the Windows console
    # defaults to cp1252 and would raise UnicodeEncodeError on every print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="python -m checker.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("courses", help="list your active courses")
    p_works = sub.add_parser("works", help="list assignments in a course")
    p_works.add_argument("--course", required=True)
    sub.add_parser("status", help="check Ollama and the code runner")
    sub.add_parser("models", help="list the LLMs installed on this machine")
    p_key = sub.add_parser("answerkey", help="condense a model answer sheet into a short key")
    p_key.add_argument("--file", required=True, help="the model answer sheet (.md, .txt, .ipynb)")
    p_key.add_argument("--model", default="", help="chat model to use")
    p_key.add_argument("--no-llm", action="store_true", help="pattern matching only")
    p_check = sub.add_parser("check", help="run checks on one assignment")
    p_check.add_argument("--course", required=True)
    p_check.add_argument("--work", required=True)
    p_check.add_argument("--checks", default=",".join(checks.ALL_CHECKS),
                         help="comma list of: " + ", ".join(checks.ALL_CHECKS))
    p_check.add_argument("--key", help="answer key file for the numbers check")
    p_check.add_argument("--model-answer", help="teacher's model answer sheet; condensed once, "
                                                "then reused (fills --key when it is not given)")
    p_check.add_argument("--tests", help="test cases file for the code check")
    p_check.add_argument("--model", default="", help="chat model to use; see `models` "
                                                     "(default: $OLLAMA_MODEL)")
    p_check.add_argument("--embed-model", default="", help="embedding model for copy detection")
    p_check.add_argument("--no-llm", action="store_true", help="heuristics only, no Ollama calls")
    p_check.add_argument("--threshold", type=float, default=0.3, help="copy-detection threshold (0-1)")
    p_check.add_argument("--include-unsubmitted", action="store_true")
    p_check.add_argument("--no-triage", action="store_true",
                         help="skip the Dify triage agent even when DIFY_TRIAGE_KEY is set")
    p_triage = sub.add_parser("triage", help="show the follow-up list for a saved report")
    p_triage.add_argument("--report", default="latest", help="report file name, or 'latest'")
    p_triage.add_argument("--agent", action="store_true",
                          help="also run the Dify triage workflow and show its notes")
    args = parser.parse_args(argv)

    if args.cmd == "status":
        s = llm.status()
        print(f"Ollama: {'OK' if s['ok'] else 'NOT REACHABLE'} at {s['url']}")
        if s["ok"]:
            print(f"  chat model  {s['model']}: {'installed' if s['model_ok'] else 'missing -> ollama pull ' + s['model']}")
            print(f"  embed model {s['embed_model']}: {'installed' if s['embed_ok'] else 'missing -> ollama pull ' + s['embed_model']}")
            print(f"  context in use: {s['num_ctx']:,} tokens")
            print("  other models available: run `python -m checker.cli models`")
        ok, note = runner.runner_available()
        print(f"Code runner: {note}" if ok else f"Code runner: unavailable - {note}")
        return 0

    if args.cmd == "models":
        found = llm.installed()
        if not found:
            print(f"No models found. Is Ollama running at {llm.OLLAMA_URL}?")
            return 1
        print(f"{'MODEL':26} {'SIZE':>8}  {'PARAMS':>7}  {'CONTEXT':>9}  USE")
        for m in found:
            use = ", ".join(filter(None, ["chat" if m["can_chat"] else "",
                                          "embeddings" if m["can_embed"] else ""]))
            context = f"{m['context']:,}" if m["context"] else "?"
            print(f"{m['name']:26} {m['size_gb']:6.1f} GB  {m['parameters'] or '?':>7}  "
                  f"{context:>9}  {use}")
        print(f"\nIn use now: chat {llm.chat_model()}, embeddings {llm.embed_model()}.")
        print("Pick another with --model / --embed-model, or set OLLAMA_MODEL in .env.")
        return 0

    if args.cmd == "answerkey":
        with open(args.file, "rb") as fh:
            text = answerkey.read_upload(args.file, fh.read())
        llm.select(args.model)
        built = answerkey.load_or_build(text, use_llm=not args.no_llm,
                                        title=os.path.basename(args.file))
        for e in built["key"]["errors"]:
            print(f"! {e}", file=sys.stderr)
        print(built["markdown"])
        print(f"\nSaved to {built['path'][:-5]}.md"
              f"{' (reused from cache)' if built['cached'] else ''}", file=sys.stderr)
        return 0

    if args.cmd == "triage":
        try:
            path = agent_tools.resolve(args.report)
        except agent_tools.ToolError as e:
            print(e, file=sys.stderr)
            return 1
        report = agent_tools.load_report(path)
        print(triage.with_names(triage.brief_markdown(triage.brief(report, os.path.basename(path))), report))
        if args.agent:
            return _run_agent(path, report)
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
                              model=args.model, embed_model=args.embed_model,
                              answer_key=read_arg_file(args.key),
                              model_answer=read_arg_file(args.model_answer),
                              tests=read_arg_file(args.tests),
                              dup_threshold=args.threshold,
                              include_unsubmitted=args.include_unsubmitted)
        bar = Bar()
        try:
            report = checks.run(cr, args.course, args.work, opts,
                                progress=checks.Progress(bar))
        finally:
            bar.close()
        print_summary(report)
        print_scores(report)
        path = checks.save(report)
        csv_path = results.save_csv(report, path[:-len(".json")] + ".csv")
        print(f"\nFull report: {path}")
        print(f"Results CSV: {csv_path}")
        if dify.configured() and not args.no_triage:
            return _run_agent(path, report)
    return 0


def _run_agent(path: str, report: dict) -> int:
    if not dify.configured():
        print("\nSet DIFY_TRIAGE_KEY (and DIFY_API_URL) to use the triage agent.", file=sys.stderr)
        return 1
    print("\nAsking the triage agent (Dify)...", file=sys.stderr)
    outcome = dify.run_triage(path)
    if outcome.get("state") != "done":
        print(f"! Triage failed: {outcome.get('error')}", file=sys.stderr)
        return 1
    print("\n-- Agent triage --")
    print(triage.with_names(outcome.get("text", ""), report))
    if outcome.get("hold"):
        print("\nSuggested to hold: " + triage.with_names(", ".join(outcome["hold"]), report))
    print(f"Review and approve draft grades: {agent_tools.report_url(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
