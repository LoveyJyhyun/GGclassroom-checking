"""Run the selected checks over every submission of one assignment.

Shared by the web app and the command-line tool.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import aidetect, duplicates, llm, numbers, runner
from . import status as st
from .classroom import Classroom

ALL_CHECKS = ("duplicates", "numbers", "ai", "code")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")


@dataclass
class Options:
    checks: tuple[str, ...] = ALL_CHECKS
    use_llm: bool = True
    answer_key: str = ""
    tests: str = ""
    dup_threshold: float = 0.3
    llm_review_pairs: int = 5
    include_unsubmitted: bool = False


@dataclass
class StudentWork:
    user_id: str
    name: str
    email: str
    status: str
    submission_id: str
    texts: list[tuple[str, str]] = field(default_factory=list)          # (title, text)
    code: list[tuple[str, str, str]] = field(default_factory=list)      # (title, language, source)
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(t for _, t in self.texts)

    @property
    def all_code(self) -> tuple[str, str] | None:
        if not self.code:
            return None
        language = self.code[0][1]
        return language, "\n\n".join(src for _, lang, src in self.code if lang == language)


def assignment_brief(work: dict) -> str:
    return f"{work.get('title', '')}\n{work.get('description', '')}".strip()


def collect(cr: Classroom, course_id: str, work: dict, include_unsubmitted: bool = False,
            progress: Callable[[str], None] = lambda _: None) -> list[StudentWork]:
    students = {s["userId"]: s for s in cr.students(course_id)}
    due = st.due_datetime(work)
    result = []
    for sub in cr.submissions(course_id, work["id"]):
        stu = students.get(sub["userId"])
        if stu is None:
            continue
        status = st.submission_status(sub, due)
        if status not in st.DONE and not include_unsubmitted:
            continue
        prof = stu["profile"]
        sw = StudentWork(sub["userId"], prof["name"].get("fullName", sub["userId"]),
                         prof.get("emailAddress", ""), status, sub["id"])
        progress(f"Reading {sw.name}")
        answer = st.answer_text(sub)
        if isinstance(answer, str) and answer.strip():
            sw.texts.append(("Answer", answer))
        for att in st.attachments_of(sub):
            if not att["file_id"]:
                if att["kind"] != "link":
                    sw.notes.append(f"{att['title']}: {att['kind']} attachments are not checked.")
                continue
            info = cr.file_content(att["file_id"])
            title = info["name"] or att["title"]
            if info["text"] is None:
                sw.notes.append(f"{title}: {info['note']}")
                continue
            language = runner.language_of(title, info["mime"])
            if language:
                source = info["text"]
                if title.lower().endswith(".ipynb") or "colaboratory" in info["mime"]:
                    try:
                        source = runner.notebook_to_script(source)
                    except ValueError:
                        sw.notes.append(f"{title}: notebook could not be parsed.")
                        continue
                sw.code.append((title, language, source))
            else:
                sw.texts.append((title, info["text"]))
        result.append(sw)
    result.sort(key=lambda s: s.name.lower())
    return result


def run(cr: Classroom, course_id: str, work_id: str, opts: Options,
        progress: Callable[[str], None] = lambda _: None) -> dict:
    work = cr.one_course_work(course_id, work_id)
    brief = assignment_brief(work)
    progress("Collecting submissions")
    people = collect(cr, course_id, work, opts.include_unsubmitted, progress)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "course_id": course_id,
        "work": {"id": work_id, "title": work.get("title", ""),
                 "link": work.get("alternateLink", "")},
        "options": {"checks": list(opts.checks), "use_llm": opts.use_llm,
                    "model": llm.OLLAMA_MODEL if opts.use_llm else None},
        "students": {p.user_id: {"name": p.name, "email": p.email, "status": p.status,
                                 "submission_id": p.submission_id, "notes": p.notes,
                                 "files": [t for t, _ in p.texts] + [t for t, _, _ in p.code]}
                     for p in people},
        "duplicates": None, "numbers": None, "ai": None, "code": None, "errors": [],
    }

    if "duplicates" in opts.checks:
        progress("Comparing submissions between students")
        result = duplicates.find(
            {p.user_id: p.text for p in people},
            {p.user_id: p.all_code for p in people if p.all_code},
            threshold=opts.dup_threshold, use_embeddings=opts.use_llm,
            llm_review=opts.llm_review_pairs if opts.use_llm else 0, assignment=brief)
        report["duplicates"] = result["pairs"]
        report["errors"] += result["errors"]

    if "numbers" in opts.checks:
        key = numbers.parse_key(opts.answer_key)
        if not key:
            report["errors"].append("Number check skipped: the answer key is empty or could not be parsed.")
        else:
            report["numbers"] = {"key": key, "results": {}}
            for p in people:
                progress(f"Checking numbers: {p.name}")
                text = "\n".join(filter(None, [p.text] + [src for _, _, src in p.code]))
                report["numbers"]["results"][p.user_id] = numbers.check(text, key, opts.use_llm)

    if "ai" in opts.checks:
        report["ai"] = {}
        for p in people:
            if not p.text.strip():
                continue
            progress(f"Checking for AI writing: {p.name}")
            report["ai"][p.user_id] = aidetect.check(p.text, brief, opts.use_llm)

    if "code" in opts.checks:
        tests = runner.parse_tests(opts.tests)
        report["code"] = {"tests_defined": len(tests), "results": {}}
        for p in people:
            if not p.all_code:
                continue
            language, source = p.all_code
            progress(f"Running code: {p.name}")
            report["code"]["results"][p.user_id] = {
                "files": [t for t, _, _ in p.code], "language": language,
                **runner.check(source, language, tests, brief, opts.use_llm)}

    return report


def save(report: dict, directory: str | None = None) -> str:
    directory = directory or REPORTS_DIR
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", report["work"]["title"]).strip("-")[:40] or "assignment"
    path = os.path.join(directory, f"{stamp}_{slug}_{report['work']['id']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return path
