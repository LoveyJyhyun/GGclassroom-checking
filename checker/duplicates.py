"""Find students whose submissions look copied from each other.

Three independent measures:
  * lexical   - overlap of 5-word sequences (catches copy-paste)
  * semantic  - embedding cosine via Ollama (catches paraphrasing)
  * code      - token overlap after renaming identifiers (catches renamed variables)
The most suspicious pairs can then be reviewed by the LLM.
"""
from __future__ import annotations

import builtins
import io
import keyword
import re
import tokenize
from itertools import combinations

from . import llm

_BUILTINS = set(dir(builtins))


def _words(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


def shingles(tokens: list[str], size: int = 5) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def normalize_python(source: str) -> list[str]:
    """Tokens with comments/whitespace removed and identifiers/literals replaced."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                            tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
                continue
            if tok.type == tokenize.NAME:
                out.append(tok.string if keyword.iskeyword(tok.string) or tok.string in _BUILTINS else "ID")
            elif tok.type == tokenize.STRING:
                out.append("STR")
            elif tok.type == tokenize.NUMBER:
                out.append("NUM")
            else:
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return normalize_generic_code(source)
    return out


def normalize_generic_code(source: str) -> list[str]:
    source = re.sub(r"//.*|#.*|/\*.*?\*/", " ", source, flags=re.DOTALL)
    tokens = re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|\S", source)
    return ["NUM" if t[0].isdigit() else t for t in tokens]


def code_tokens(source: str, language: str) -> list[str]:
    return normalize_python(source) if language == "python" else normalize_generic_code(source)


LLM_SYSTEM = (
    "You compare two students' submissions for the same assignment and judge whether one "
    "was copied from the other (directly, paraphrased, or with renamed variables). "
    "Similar answers to a narrow question are normal; look for shared unusual wording, "
    "identical mistakes, identical structure or ordering. "
    'Reply with JSON only: {"copied": true|false, "confidence": <0-100>, "evidence": ["...", ...]}'
)


def llm_compare(a: str, b: str, assignment: str = "") -> dict:
    half = llm.MAX_PROMPT_CHARS // 2 - 200
    prompt = (f"Assignment:\n{assignment or '(not provided)'}\n\n"
              f"Submission A:\n<<<\n{llm.truncate(a, half)}\n>>>\n\n"
              f"Submission B:\n<<<\n{llm.truncate(b, half)}\n>>>")
    data = llm.chat_json(LLM_SYSTEM, prompt)
    evidence = data.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    try:
        confidence = int(float(data.get("confidence", 0)))
    except (TypeError, ValueError):
        confidence = None
    return {"copied": bool(data.get("copied")), "confidence": confidence,
            "evidence": [str(e) for e in evidence][:5]}


def find(texts: dict[str, str], code: dict[str, tuple[str, str]] | None = None,
         threshold: float = 0.3, semantic_threshold: float = 0.92,
         use_embeddings: bool = True, llm_review: int = 5, assignment: str = "") -> dict:
    """Return suspicious pairs.

    texts: student id -> written text
    code:  student id -> (language, source)
    """
    code = code or {}
    errors = []
    word_sets = {k: shingles(_words(v)) for k, v in texts.items() if v and v.strip()}
    code_sets = {k: shingles(code_tokens(src, lang), 8) for k, (lang, src) in code.items() if src.strip()}

    vectors = {}
    if use_embeddings and len(word_sets) > 1:
        keys = sorted(word_sets)
        try:
            vectors = dict(zip(keys, llm.embed([texts[k] for k in keys])))
        except llm.LLMError as e:
            errors.append(f"Embeddings skipped: {e}")

    people = sorted(set(word_sets) | set(code_sets))
    pairs = []
    for a, b in combinations(people, 2):
        lexical = jaccard(word_sets[a], word_sets[b]) if a in word_sets and b in word_sets else None
        semantic = llm.cosine(vectors[a], vectors[b]) if a in vectors and b in vectors else None
        code_sim = jaccard(code_sets[a], code_sets[b]) if a in code_sets and b in code_sets else None
        flagged = ((lexical or 0) >= threshold or (code_sim or 0) >= max(threshold, 0.5)
                   or (semantic or 0) >= semantic_threshold)
        if flagged:
            pairs.append({"a": a, "b": b, "lexical": lexical, "semantic": semantic,
                          "code": code_sim, "llm": None})
    pairs.sort(key=lambda p: max(p["lexical"] or 0, p["code"] or 0, (p["semantic"] or 0) - 0.5),
               reverse=True)

    for pair in pairs[:llm_review]:
        a_text = "\n".join(filter(None, [texts.get(pair["a"]), code.get(pair["a"], ("", ""))[1]]))
        b_text = "\n".join(filter(None, [texts.get(pair["b"]), code.get(pair["b"], ("", ""))[1]]))
        try:
            pair["llm"] = llm_compare(a_text, b_text, assignment)
        except llm.LLMError as e:
            errors.append(f"LLM review skipped: {e}")
            break

    return {"pairs": pairs, "checked": len(people), "errors": errors}
