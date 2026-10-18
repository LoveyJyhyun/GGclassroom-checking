"""The tools an agent may use, shared by the MCP server and the plain JSON API.

Read tools return coded data only (see triage.py): no names, emails or answers.
The single write tool, save_triage, stores the agent's notes and a draft-grade
proposal next to the report. Nothing an agent does reaches Google Classroom:
grades are saved only when the teacher approves them on the report page.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable

from . import checks, grades, triage

APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8080").rstrip("/")
REPORT_NAME = re.compile(r"^[\w.-]+\.json$")
MAX_TRIAGE_CHARS = 20_000


class ToolError(ValueError):
    """A problem the agent can fix (wrong report name, unknown student code ...)."""


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    params: dict                      # JSON-schema properties
    required: tuple[str, ...]
    handler: Callable[..., tuple[str, dict]]
    read_only: bool = True

    def schema(self) -> dict:
        return {"type": "object", "properties": self.params,
                "required": list(self.required), "additionalProperties": False}


# --- Reports -----------------------------------------------------------------

def report_files() -> list[str]:
    """Saved reports, newest first (file names start with their timestamp)."""
    try:
        names = os.listdir(checks.REPORTS_DIR)
    except OSError:
        return []
    return sorted((n for n in names if REPORT_NAME.match(n)), reverse=True)


def resolve(report: str | None) -> str:
    """'latest', a report file name or its stem -> the report's path."""
    name = os.path.basename((report or "").strip())
    files = report_files()
    if name in ("", "latest"):
        if not files:
            raise ToolError("There are no saved reports yet. Run a check first.")
        name = files[0]
    if not name.endswith(".json"):
        name += ".json"
    if not REPORT_NAME.match(name) or name not in files:
        hint = f" The newest is {files[0]}." if files else ""
        raise ToolError(f"No report named {name!r}. Call list_reports to see them.{hint}")
    return os.path.join(checks.REPORTS_DIR, name)


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def report_url(path: str) -> str:
    return f"{APP_BASE_URL}/report/{os.path.basename(path)}#follow-up"


# --- Tool handlers -------------------------------------------------------------

def list_reports(limit: int = 10, assignment: str = "") -> tuple[str, dict]:
    try:
        limit = max(1, min(int(float(limit or 10)), 50))
    except (TypeError, ValueError):
        raise ToolError("limit must be a number.") from None
    found = []
    for name in report_files():
        path = os.path.join(checks.REPORTS_DIR, name)
        try:
            r = load_report(path)
        except (OSError, ValueError):
            continue
        title = (r.get("work") or {}).get("title", "")
        if assignment and assignment.lower() not in title.lower():
            continue
        state = (triage.load_triage(path) or {}).get("state", "none")
        found.append({"report": name, "assignment": title,
                      "course": (r.get("course") or {}).get("name", ""),
                      "checked_at": r.get("generated_at", ""),
                      "students": len(r.get("students") or {}), "triage": state})
        if len(found) >= limit:
            break
    if not found:
        return "No saved reports match.", {"reports": []}
    lines = [f"- {f['report']} · {f['assignment']} · {f['course']} · {f['checked_at'][:16]} · "
             f"{f['students']} students · triage: {f['triage']}" for f in found]
    return "Saved check reports, newest first:\n" + "\n".join(lines), {"reports": found}


def triage_brief(report: str = "latest") -> tuple[str, dict]:
    path = resolve(report)
    b = triage.brief(load_report(path), os.path.basename(path))
    return triage.brief_markdown(b) + f"\n\nReport file: {b['report']}", b


def student_detail(report: str = "latest", student: str = "") -> tuple[str, dict]:
    path = resolve(report)
    detail = triage.student_detail(load_report(path), student or "")
    if detail is None:
        raise ToolError(f"No student {student!r} in this report. Use a code from triage_brief, like S01.")
    return json.dumps(detail, ensure_ascii=False, indent=1), detail


def save_triage(report: str, triage_text: str, hold: str = "") -> tuple[str, dict]:
    path = resolve(report)
    text = (triage_text or "").strip()
    if not text:
        raise ToolError("triage is empty.")
    text = text[:MAX_TRIAGE_CHARS]
    r = load_report(path)
    hold_codes = triage.parse_codes(hold, r)
    triage.write_triage(path, state="done", text=text, hold=hold_codes, error=None, source="agent")

    note = "No draft grades proposed: this run has no scores."
    proposal_info: dict = {}
    if any(grades.scores_of(r).values()):
        # The reason the teacher sees comes from the report, not from the agent.
        b = triage.brief(r)
        code_to_uid = {c: u for u, c in triage.codes(r).items()}
        first_signal = {p["code"]: p["signals"][0]["text"] for p in b["students"] if p["signals"]}
        held = {code_to_uid[c]: grades.AGENT_HOLD + first_signal.get(c, "see the triage")
                for c in hold_codes}
        with grades.lock_for(path):
            existing = grades.load(path)
            if grades.is_locked(existing):
                proposal = None
                note = "Draft grades were already saved by the teacher, so the proposal was left as it is."
            elif existing:
                # Keep the teacher's points and choices; only update the agent's holds.
                proposal = grades.merge_agent_holds(existing, held)
                grades.save(path, proposal)
            else:
                try:
                    proposal = grades.propose(r, hold=held, created_by="agent")
                    grades.save(path, proposal)
                except ValueError:
                    proposal = None
                    note = "No draft grades proposed: the assignment has no point value."
        if proposal:
            proposal_info = {"proposed": len(proposal["rows"]), "held": hold_codes,
                             "can_save_to_classroom": proposal["can_save"]}
            note = (f"Draft grades proposed for {len(proposal['rows'])} students"
                    + (f", {len(hold_codes)} held ({', '.join(hold_codes)})" if hold_codes else "")
                    + ". Nothing is sent to Classroom until the teacher approves them.")
    url = report_url(path)
    return (f"Saved the triage. {note}\nTeacher review page: {url}",
            {"saved": True, "url": url, "hold": hold_codes, "proposal": proposal_info})


def grade_proposal(report: str = "latest") -> tuple[str, dict]:
    path = resolve(report)
    proposal = grades.load(path)
    if not proposal:
        return "No draft-grade proposal for this report yet.", {"proposal": None}
    code_of = triage.codes(load_report(path))
    rows = [{"code": code_of.get(row["uid"], "?"), "percent": row["percent"], "grade": row["grade"],
             "held": row["held"], "result": row.get("result")} for row in proposal["rows"]]
    lines = [f"Status: {proposal['status']} · out of {proposal['max_points']} points"]
    lines += [f"- {x['code']}: {x['percent']}% -> {x['grade']}" + (" (held)" if x["held"] else "")
              + (f" [{x['result']}]" if x["result"] else "") for x in rows]
    return "\n".join(lines), {"status": proposal["status"], "max_points": proposal["max_points"], "rows": rows}


REPORT_PARAM = {"type": "string", "description": "Report file name from list_reports, or 'latest'.",
                "default": "latest"}

TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("list_reports", "List check reports",
         "List saved check reports, newest first, with whether each one has been triaged.",
         {"limit": {"type": "integer", "description": "How many to list (max 50).", "default": 10},
          "assignment": {"type": "string", "description": "Only reports whose assignment title contains this."}},
         (), list_reports),
    Tool("triage_brief", "Triage brief for a report",
         "Facts for triaging one check run: class summary, then every student as a code (S01 ...) "
         "with a tier (Needs action / Worth a look / No concerns) and the signals behind it. "
         "Copying and AI-writing signals are leads to follow up, never proof.",
         {"report": REPORT_PARAM}, (), triage_brief),
    Tool("student_detail", "One student's results",
         "Marks per question, missing points, code test results and every signal for one student code.",
         {"report": REPORT_PARAM,
          "student": {"type": "string", "description": "Student code from triage_brief, e.g. S03."}},
         ("student",), student_detail),
    Tool("save_triage", "Save the triage for the teacher",
         "Store your triage notes on the report page (replacing earlier notes) and propose draft "
         "grades from the scores. List in 'hold' the codes whose grade the teacher should check before "
         "saving it (e.g. likely copying, work cut short). This does not change anything in Google "
         "Classroom, and never touches grades the teacher has already saved.",
         {"report": REPORT_PARAM,
          "triage": {"type": "string", "description": "Your triage in Markdown. Refer to students by code only."},
          "hold": {"type": "string", "description": "Codes to hold back, e.g. 'S03, S07'. Empty for none."}},
         ("triage",), lambda report="latest", triage="", hold="": save_triage(report, triage, hold),
         read_only=False),
    Tool("grade_proposal", "Current draft-grade proposal",
         "The proposed draft grades for a report (by code) and whether the teacher has saved them.",
         {"report": REPORT_PARAM}, (), grade_proposal),
]}


def call(name: str, arguments: dict | None) -> tuple[str, dict]:
    """Run one tool. Raises KeyError for an unknown tool, ToolError for bad input."""
    tool = TOOLS[name]
    if arguments is not None and not isinstance(arguments, dict):
        raise ToolError("Arguments must be a JSON object.")
    args = dict(arguments or {})
    unknown = set(args) - set(tool.params)
    if unknown:
        raise ToolError(f"Unknown argument(s): {', '.join(sorted(unknown))}.")
    for key, value in args.items():
        kind = tool.params[key]["type"]
        if kind == "string" and value is not None and not isinstance(value, str):
            raise ToolError(f"{key} must be a string.")
        if kind == "integer" and isinstance(value, (bool, dict, list)):
            raise ToolError(f"{key} must be a number.")
    missing = [p for p in tool.required if not str(args.get(p) or "").strip()]
    if missing:
        raise ToolError(f"Missing argument(s): {', '.join(missing)}.")
    return tool.handler(**args)
