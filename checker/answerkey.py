"""Condense a teacher's model answer sheet into a compact, reusable answer key.

A model answer sheet ("dap an mau") is usually a long document. It is distilled
**once**, cached by content hash, and written out as markdown: numeric answers in
the plain ``label: value ± tol`` format that :func:`numbers.parse_key` already
reads, and the required points of written answers as short bullets.

Every later comparison therefore works from the small key:

* numeric answers are compared in Python, with no LLM call at all;
* written answers send only the point list (a few words each), never the sheet.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from . import documents, llm, numbers, runner

KEYS_DIR = os.environ.get("ANSWER_KEYS_DIR", "answer_keys")
FORMAT_VERSION = "1"
MAX_POINTS = 6

DISTILL_SYSTEM = (
    "You condense a teacher's MODEL ANSWER SHEET into a compact answer key. "
    "Copy the teacher's answers exactly; never solve, re-derive or correct anything. "
    "For a numeric answer, give the number as the sheet writes it, plus its tolerance if "
    "the sheet states one. For a written answer, give the few points a correct answer must "
    "contain, as short phrases (not sentences). Keep the sheet's own question labels. "
    'Reply with JSON only: {"questions": [{"label": "1", "answer": "42", "tol": "0.01", '
    '"points": ["short phrase", "..."]}]}'
)

GRADE_SYSTEM = (
    "You check one student's answer against a list of REQUIRED POINTS, each with an id. "
    "A point counts as present if the student's own words convey it, even when worded "
    "differently, negated the other way round, or split across sentences. "
    "Judge nothing else: not style, not spelling, not points outside the list. "
    "Write the note in the same language as the student's answer. "
    'Reply with JSON only: {"missing": ["<id of each point the student does not convey>"], '
    '"note": "one short sentence"}'
)



# --- building the key -------------------------------------------------------

# A sheet with many questions needs a longer reply than a single judgement does.
DISTILL_REPLY_TOKENS = int(os.environ.get("OLLAMA_DISTILL_REPLY_TOKENS", "1600"))


def _chunks(text: str, limit: int | None = None) -> list[str]:
    """Split a long sheet on blank lines, so no question is cut in half.

    Without this, a long sheet would be silently head+tail truncated by the LLM
    client and the middle questions would vanish from the key.
    """
    if limit is None:
        limit = llm.max_user_chars(DISTILL_SYSTEM, DISTILL_REPLY_TOKENS) - 200
    if len(text) <= limit:
        return [text]
    out, buf = [], ""
    for para in text.split("\n\n"):
        if buf and len(buf) + len(para) + 2 > limit:
            out.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        out.append(buf)
    return out


def _looks_like_an_answer(raw: str) -> bool:
    """An answer-key line is mostly its value; a sentence is mostly words.

    Without this, pattern mode reads "Câu 1. Tính tổng hai số 20 và 22." as the
    answer 22 - a number lifted straight out of the question - and the LLM's prose
    answers ("205 dòng x 9 cột; thu_nhap thiếu 14") get marked as single numbers.
    """
    # ponytail: word count, not grammar. Raise the limit if real keys get dropped.
    words = [w for w in re.findall(r"[^\W\d_]+", raw or "", re.UNICODE) if len(w) > 1]
    return len(words) <= 3


def is_plain_number(answer: str) -> bool:
    """True only when the whole answer is a value, not a sentence containing one."""
    return numbers.last_number(answer) is not None and _looks_like_an_answer(answer)


def _distill_chunk(part: str) -> dict:
    return llm.chat_json(DISTILL_SYSTEM, f"Model answer sheet:\n{part}",
                         reply_tokens=DISTILL_REPLY_TOKENS)


def _split_on_failure(parts: list[str], errors: list[str], depth: int = 2):
    """Condense each chunk; on failure, halve it and try the halves.

    A small model asked about a long, varied sheet can loop ("token repeat limit
    reached") or run past the token limit. Retrying the same text at temperature 0
    gives the same failure, so the retry has to make the input smaller.
    """
    for part in parts:
        try:
            yield _distill_chunk(part)
            continue
        except llm.LLMError as e:
            paragraphs = part.split("\n\n")
            if depth <= 0 or len(paragraphs) < 2:
                errors.append(f"Part of the model answer could not be condensed: {e}")
                continue
        middle = len(paragraphs) // 2
        halves = ["\n\n".join(paragraphs[:middle]), "\n\n".join(paragraphs[middle:])]
        yield from _split_on_failure(halves, errors, depth - 1)


def _clean(raw: dict) -> dict | None:
    label = str(raw.get("label", "")).strip().lower()
    if not label:
        return None
    answer = raw.get("answer")
    answer = "" if answer is None else str(answer).strip()
    tol = raw.get("tol")
    tol = str(tol).strip() if tol not in (None, "", "null", "None") else None
    points = raw.get("points") or []
    if isinstance(points, str):
        points = [points]
    points = [str(p).strip() for p in points if str(p).strip()][:MAX_POINTS]

    # Small models invent a "± 0" tolerance the sheet never stated; the default
    # 0.1% relative tolerance is a better answer than demanding exact equality.
    if tol is not None and not (numbers.parse_number(tol.lstrip("±+-/ ")) or 0):
        tol = None
    # They also echo the number back as a "required point". Numeric answers are
    # compared exactly in Python, so those points are noise and cost tokens.
    # Only for real numbers: a prose answer that happens to contain a number keeps
    # its points, because that is the only way it can be marked at all.
    if points and is_plain_number(answer):
        points = []

    if not answer and not points:
        return None
    return {"label": label, "answer": answer, "tol": tol, "points": points}


def _sort_label(q: dict):
    m = re.match(r"(\d+)(.*)", q["label"])
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, q["label"])


def is_numeric(q: dict) -> bool:
    """True when the answer line should be read back as a number by parse_key."""
    return is_plain_number(q["answer"])


def distill(text: str, use_llm: bool = True) -> dict:
    """Turn a model answer sheet into {"questions": [...], "source", "errors"}."""
    if not text.strip():
        return {"questions": [], "source": "empty", "errors": ["The model answer file is empty."]}

    if not use_llm:
        found = [{"label": lab, "answer": v["raw"] or f'{v["value"]:g}',
                  "tol": v["tol"], "points": []}
                 for lab, v in numbers.parse_key(text).items()
                 if _looks_like_an_answer(v["raw"])]
        return {"questions": found, "source": "pattern",
                "errors": [] if found else [
                    "Without the LLM only lines like `1: 42` or `Câu 1: 42` are read, and the "
                    "sheet has none. Run without --no-llm to summarise a worked answer sheet."]}

    questions: dict[str, dict] = {}
    errors: list[str] = []
    for part in _split_on_failure(_chunks(text), errors):
        for raw in (part.get("questions") or []):
            if isinstance(raw, dict):
                q = _clean(raw)
                if q:
                    questions.setdefault(q["label"], q)      # first mention wins
    # "2" -> "7: 5" means the model put a sub-question number where the answer goes,
    # and the key would then mark question 2 against 5. Cannot be guessed safely, so
    # say which ones to look at.
    # Note the separator excludes ".", or every decimal ("12.5") would look like a label.
    confused = [q["label"] for q in questions.values()
                if re.match(r"^\s*\d+\s*[:)]\s", q["answer"])]
    if confused:
        errors.append(
            "Check these answers in the generated key by hand - they still contain a "
            f"question number, so the value may be wrong: {', '.join(sorted(confused)[:12])}.")

    if not questions:
        fallback = distill(text, use_llm=False)
        if fallback["questions"]:
            fallback["errors"] = errors + ["The LLM returned no questions; used pattern matching."]
            return fallback
        errors.append("No questions could be read from the model answer file.")
    return {"questions": sorted(questions.values(), key=_sort_label),
            "source": "llm", "errors": errors}


def to_markdown(key: dict, title: str = "") -> str:
    """Render the key as markdown whose numeric lines parse_key can read back."""
    head = f"# Answer key{' - ' + title if title else ''}"
    out = [head, "",
           f"<!-- Condensed from the model answer sheet ({key['source']}). One answer per line,"
           " format `label: value ± tolerance`. Edit freely; this file is what the checks read. -->",
           ""]

    numeric = [q for q in key["questions"] if is_numeric(q)]
    if numeric:
        out.append("## Numeric answers")
        out.append("")
        for q in numeric:
            line = f'{q["label"]}: {q["answer"]}'
            if q["tol"] and "±" not in q["answer"]:
                line += f' ± {q["tol"].lstrip("±+-/ ").strip()}'
            out.append(line)
        out.append("")

    written = [q for q in key["questions"] if q["points"]]
    if written:
        out.append("## Required points for written answers")
        out.append("")
        for q in written:
            # The leading "- " keeps parse_key from reading these as numeric answers.
            out.append(f'- {q["label"]}: ' + "; ".join(q["points"]))
        out.append("")

    # Answers that are neither a plain number nor a point list still belong in the
    # file - dropping them would silently lose part of the teacher's sheet.
    other = [q for q in key["questions"] if not is_numeric(q) and not q["points"] and q["answer"]]
    if other:
        out.append("## Not marked automatically")
        out.append("")
        out.append("<!-- Too long to compare as a number and with no point list, so these are")
        out.append("     for reference. Rewrite one as `label: value` or as `- label: point; point`")
        out.append("     to have it marked. -->")
        out.append("")
        for q in other:
            out.append(f'- {q["label"]}: {q["answer"]}')
        out.append("")
    return "\n".join(out)


def _digest(text: str, use_llm: bool) -> str:
    # The model is part of the key: a different model condenses the sheet differently,
    # so switching model must build a new key instead of reusing the old one.
    raw = f"{FORMAT_VERSION}|{use_llm}|{llm.chat_model() if use_llm else ''}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_or_build(text: str, use_llm: bool = True, title: str = "",
                  directory: str | None = None) -> dict:
    """Distill the sheet, or reuse the cached key if this exact sheet was seen before.

    Content-addressed: re-uploading the same file costs nothing.
    """
    directory = directory or KEYS_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{_digest(text, use_llm)}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return {**json.load(fh), "path": path, "cached": True}

    key = distill(text, use_llm)
    built = {"key": key, "markdown": to_markdown(key, title)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(built, fh, ensure_ascii=False, indent=2)
    with open(path[:-5] + ".md", "w", encoding="utf-8") as fh:
        fh.write(built["markdown"])
    return {**built, "path": path, "cached": False}


def read_upload(filename: str, data: bytes) -> str:
    """Decode an uploaded model answer file to text.

    Whatever a submission can be read from, an answer sheet can be read from too:
    .md, .txt, .docx, .xlsx, .pptx, a notebook, or a .zip holding any of those.
    """
    found = documents.extract(filename, data)
    if not found["parts"]:
        raise ValueError(
            f"{filename} could not be read. {found['note']} "
            "Save the model answer as .md, .txt or .docx, or paste its text below.")

    text = "\n\n".join(part for _, part in found["parts"])
    if filename.lower().endswith(".ipynb"):
        try:
            text = runner.notebook_to_script(text)
        except ValueError:
            pass
    return text


# --- comparing a student against the key ------------------------------------

def _normalise(s: str) -> str:
    """Lowercase, collapse everything that is not a word character, pad with spaces."""
    return " " + re.sub(r"\W+", " ", s.lower(), flags=re.UNICODE).strip() + " "


def grade(text: str, questions: list[dict]) -> dict:
    """Compare one student's text against the required points.

    A point the student states almost word for word is settled here, for free.
    Only the undecided points go to the LLM, as a numbered list, and it replies
    with the ids it cannot find - so the whole exchange is a handful of tokens.
    """
    items = [(f'{q["label"]}.{i}', q["label"], p)
             for q in questions for i, p in enumerate(q["points"], 1)]
    if not items:
        return {"results": [], "covered": 0, "total": 0, "asked": 0, "note": "", "error": None}

    # ponytail: exact phrase match, so word order and negation still have to line
    # up. Paraphrases fall through to the LLM, which is what it is good at.
    said = _normalise(text)
    covered_ids = {pid for pid, _, point in items if _normalise(point) in said}
    ask = [(pid, point) for pid, _, point in items if pid not in covered_ids]

    missing_ids, note, error, trimmed = set(), "", None, []
    if ask:
        listing = "\n".join(f"{pid} {p}" for pid, p in ask)
        try:
            # The point list is short, so pack() keeps it whole and trims only the
            # student's answer if it is very long.
            data, trimmed = llm.pack_json(GRADE_SYSTEM, [("Required points", listing),
                                                         ("Student answer", text)])
            missing_ids = {str(x).strip() for x in (data.get("missing") or [])}
            note = str(data.get("note", ""))[:300]
        except llm.LLMError as e:
            error = str(e)
            missing_ids = {pid for pid, _ in ask}      # undecided, so not credited

    per: dict[str, dict] = {}
    for pid, label, point in items:
        entry = per.setdefault(label, {"question": label, "covered": [], "missing": []})
        entry["missing" if pid in missing_ids else "covered"].append(point)
    results = []
    for entry in per.values():
        total = len(entry["covered"]) + len(entry["missing"])
        results.append({**entry, "total": total,
                        "score": round(100 * len(entry["covered"]) / total) if total else None})
    return {"results": results, "covered": sum(len(r["covered"]) for r in results),
            "total": len(items), "asked": len(ask), "note": note, "error": error,
            "truncated": trimmed}


def demo() -> None:
    """Self-check: the generated markdown must round-trip through parse_key."""
    key = {"source": "llm", "errors": [], "questions": [
        {"label": "1", "answer": "42", "tol": None, "points": []},
        {"label": "2", "answer": "3.14", "tol": "0.01", "points": []},
        {"label": "3", "answer": "", "tol": None,
         "points": ["conservation of energy", "units given"]},
    ]}
    md = to_markdown(key, "Lab 1")
    parsed = numbers.parse_key(md)
    assert set(parsed) == {"1", "2"}, parsed             # written answers stay out
    assert parsed["1"]["value"] == 42
    assert parsed["2"]["value"] == 3.14 and parsed["2"]["tol"] == "0.01"
    assert "- 3: conservation of energy; units given" in md
    print(md)
    print("OK: markdown round-trips through numbers.parse_key")


if __name__ == "__main__":
    demo()
