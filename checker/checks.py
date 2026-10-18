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

from . import aidetect, answerkey, duplicates, llm, numbers, results, runner
from . import status as st
from .classroom import Classroom

# Run order is fixed and meaningful: cheap whole-class comparison first, then the
# marking, then the integrity signal, then the slowest step (starting containers).
ALL_CHECKS = ("duplicates", "numbers", "rubric", "ai", "code")
PHASE_LABELS = {
    "collect": "Reading submissions",
    "answer_key": "Condensing the model answer",
    "duplicates": "1. Copying between students",
    "numbers": "2. Answers against the key",
    "rubric": "2. Written answers against the key",
    "ai": "3. AI-written text",
    "code": "4. Running code",
}
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")


class Progress:
    """Counts a run's steps so the caller can draw a progress bar.

    `total` is 0 until plan() is called, which the web UI renders as an
    indeterminate bar - the student count is not known until they are fetched.
    """

    def __init__(self, callback: Callable[[str, str, int, int], None] | None = None):
        self._cb = callback
        self.phase = ""
        self.message = ""
        self.done = 0
        self.total = 0

    def plan(self, total: int) -> None:
        self.total = max(1, total)

    def start(self, phase: str) -> None:
        self.phase = PHASE_LABELS.get(phase, phase)
        self.message = ""
        self._emit()

    def step(self, message: str = "") -> None:
        self.done = min(self.total, self.done + 1) if self.total else self.done + 1
        self.message = message
        self._emit()

    def sync(self, done: int) -> None:
        """Jump the counter to a phase boundary, so the bar cannot drift."""
        self.done = min(self.total, done) if self.total else done
        self._emit()

    def _emit(self) -> None:
        if self._cb:
            self._cb(self.phase, self.message, self.done, self.total)


@dataclass
class Options:
    checks: tuple[str, ...] = ALL_CHECKS
    use_llm: bool = True
    model: str = ""               # "" = whatever OLLAMA_MODEL is set to
    embed_model: str = ""
    answer_key: str = ""
    model_answer: str = ""        # raw text of the teacher's model answer sheet
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
            progress: Progress | None = None) -> list[StudentWork]:
    pr = progress or Progress()
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
        pr.step(sw.name)
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
            if info["note"]:                     # read, but something inside was skipped
                sw.notes.append(f"{title}: {info['note']}")

            # An archive comes back as one part per file inside it, so each member is
            # routed on its own name: a .py in a zip still reaches the code check.
            for member, text in info.get("parts") or [(title, info["text"])]:
                label = member if member != title else title
                language = runner.language_of(member, info["mime"])
                if language:
                    source = text
                    if member.lower().endswith(".ipynb") or "colaboratory" in info["mime"]:
                        try:
                            source = runner.notebook_to_script(source)
                        except ValueError:
                            sw.notes.append(f"{label}: notebook could not be parsed.")
                            continue
                    sw.code.append((label, language, source))
                else:
                    sw.texts.append((label, text))
        result.append(sw)
    result.sort(key=lambda s: s.name.lower())
    return result


def _plan(opts: Options, people: list[StudentWork]) -> list[tuple[str, int]]:
    """How many steps each selected check will take, in run order."""
    with_text = sum(1 for p in people if p.text.strip())
    steps = {
        "duplicates": 1 + (opts.llm_review_pairs if opts.use_llm else 0),
        "numbers": len(people),
        "rubric": with_text,
        "ai": with_text,
        "code": sum(1 for p in people if p.all_code),
    }
    if not opts.model_answer.strip():
        steps["rubric"] = 0          # run() skips it entirely without a sheet
    plan = []
    if opts.model_answer.strip():
        plan.append(("answer_key", 1))
    plan += [(name, steps[name]) for name in ALL_CHECKS if name in opts.checks]
    return plan


def run(cr: Classroom, course_id: str, work_id: str, opts: Options,
        progress: Progress | None = None) -> dict:
    pr = progress or Progress()
    # Pick the models before anything calls out to Ollama. llm.select is per-thread,
    # so two runs with different models cannot interfere.
    llm.select(opts.model, opts.embed_model)
    work = cr.one_course_work(course_id, work_id)
    brief = assignment_brief(work)

    pr.start("collect")
    people = collect(cr, course_id, work, opts.include_unsubmitted, pr)

    # Only now is the class size known, so the bar is indeterminate until here.
    plan = _plan(opts, people)
    collected = pr.done
    pr.plan(collected + sum(n for _, n in plan))
    base = collected

    course = cr.course(course_id)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "course_id": course_id,
        "course": {"id": course_id, "name": course.get("name", "")},
        "work": {"id": work_id, "title": work.get("title", ""),
                 "link": work.get("alternateLink", ""),
                 # For draft-grade proposals: the points the score is scaled to, and
                 # whether Google will let this app write grades on this assignment.
                 "max_points": work.get("maxPoints"),
                 "associated_with_developer": bool(work.get("associatedWithDeveloper"))},
        "options": {"checks": list(opts.checks), "use_llm": opts.use_llm,
                    "model": llm.chat_model() if opts.use_llm else None,
                    "embed_model": llm.embed_model() if opts.use_llm else None,
                    "num_ctx": llm.num_ctx_for(llm.chat_model()) if opts.use_llm else None},
        "students": {p.user_id: {"name": p.name, "email": p.email, "status": p.status,
                                 "submission_id": p.submission_id, "notes": p.notes,
                                 "files": [t for t, _ in p.texts] + [t for t, _, _ in p.code]}
                     for p in people},
        "answer_key": None,
        "duplicates": None, "numbers": None, "rubric": None, "ai": None, "code": None,
        "scores": None,
        "errors": [],
    }

    def finished(name: str) -> None:
        """Jump to this phase's planned end, so earlier estimates cannot skew the bar."""
        nonlocal base
        base += dict(plan).get(name, 0)
        pr.sync(base)

    # The model answer sheet is condensed once (and cached), never sent per student.
    answer_key, rubric_questions = opts.answer_key, []
    if opts.model_answer.strip():
        pr.start("answer_key")
        try:
            built = answerkey.load_or_build(opts.model_answer, opts.use_llm, work.get("title", ""))
            report["answer_key"] = {
                "file": os.path.basename(built["path"]), "markdown": built["markdown"],
                "source": built["key"]["source"], "cached": built["cached"],
                "questions": len(built["key"]["questions"]),
            }
            report["errors"] += built["key"]["errors"]
            if not answer_key.strip():
                answer_key = built["markdown"]
            rubric_questions = [q for q in built["key"]["questions"] if q["points"]]
        except (llm.LLMError, OSError) as e:
            report["errors"].append(f"The model answer could not be condensed: {e}")
        if not {"numbers", "rubric"} & set(opts.checks):
            report["errors"].append(
                "No score was worked out: a model answer sheet was given, but neither "
                "'Number answers' nor 'Compare with the model answer' was ticked.")
        finished("answer_key")

    # --- 1. Copying between students ---------------------------------------
    if "duplicates" in opts.checks:
        pr.start("duplicates")
        pr.step("Comparing every pair")
        result = duplicates.find(
            {p.user_id: p.text for p in people},
            {p.user_id: p.all_code for p in people if p.all_code},
            threshold=opts.dup_threshold, use_embeddings=opts.use_llm,
            llm_review=opts.llm_review_pairs if opts.use_llm else 0, assignment=brief,
            on_review=lambda a, b: pr.step(f"Reviewing {a} and {b}"))
        report["duplicates"] = result["pairs"]
        report["errors"] += result["errors"]
        finished("duplicates")

    # --- 2. Answers against the key ----------------------------------------
    if "numbers" in opts.checks:
        pr.start("numbers")
        key = numbers.parse_key(answer_key)
        if not key:
            report["errors"].append("Number check skipped: the answer key is empty or could not be parsed.")
        else:
            report["numbers"] = {"key": key, "results": {}}
            for p in people:
                pr.step(p.name)
                text = "\n".join(filter(None, [p.text] + [src for _, _, src in p.code]))
                report["numbers"]["results"][p.user_id] = numbers.check(text, key, opts.use_llm)
        finished("numbers")

    if "rubric" in opts.checks and opts.model_answer.strip():
        pr.start("rubric")
        if not rubric_questions:
            report["errors"].append(
                "Model answer comparison skipped: the sheet has no written-answer points "
                "(numeric answers are compared by the number check instead).")
        elif not opts.use_llm:
            report["errors"].append("Model answer comparison skipped: it needs the LLM.")
        else:
            report["rubric"] = {"questions": rubric_questions, "results": {}}
            for p in people:
                if not p.text.strip():
                    continue
                pr.step(p.name)
                report["rubric"]["results"][p.user_id] = answerkey.grade(p.text, rubric_questions)
        finished("rubric")

    # --- 3. AI-written text ------------------------------------------------
    if "ai" in opts.checks:
        pr.start("ai")
        report["ai"] = {}
        for p in people:
            if not p.text.strip():
                continue
            pr.step(p.name)
            report["ai"][p.user_id] = aidetect.check(p.text, brief, opts.use_llm)
        finished("ai")

    # --- 4. Running code ---------------------------------------------------
    if "code" in opts.checks:
        pr.start("code")
        tests = runner.parse_tests(opts.tests)
        report["code"] = {"tests_defined": len(tests), "results": {}}
        for p in people:
            if not p.all_code:
                continue
            language, source = p.all_code
            pr.step(p.name)
            report["code"]["results"][p.user_id] = {
                "files": [t for t, _, _ in p.code], "language": language,
                **runner.check(source, language, tests, brief, opts.use_llm)}
        finished("code")

    report["scores"] = results.scores(report)
    report["question_labels"] = results.question_labels(report)
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
