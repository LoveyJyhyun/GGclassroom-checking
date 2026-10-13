"""Minimal client for a local Ollama server (e.g. Qwen models).

Uses only the standard library so no extra dependency is needed.
"""
from __future__ import annotations

import contextvars
import json
import math
import os
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
# nomic-embed-text only has 2048 tokens, far less than the chat model.
OLLAMA_EMBED_NUM_CTX = int(os.environ.get("OLLAMA_EMBED_NUM_CTX", "2048"))

# Ollama does not refuse an over-long prompt: it drops tokens from the START, which
# eats the system prompt and the "reply with JSON" instruction with it, and the model
# then answers with prose. So every prompt is measured and packed before it is sent.
#
# Measured on qwen2.5:7b: 5.2 chars/token for English, 3.9 for Vietnamese, 3.3 for
# source code. 3.0 is the safe floor for mixed content.
CHARS_PER_TOKEN = float(os.environ.get("OLLAMA_CHARS_PER_TOKEN", "3.0"))
# Room kept for the model's own answer, and for the chat template around the messages.
REPLY_TOKENS = int(os.environ.get("OLLAMA_REPLY_TOKENS", "700"))
TEMPLATE_TOKENS = 48

TRUNCATED_MARK = "\n...[truncated]...\n"


class LLMError(RuntimeError):
    pass


# --- Which model this run uses ----------------------------------------------

# A run can pick any model installed on the machine. The choice lives in a
# ContextVar rather than a module global: the web app runs each check in its own
# thread, and two teachers picking different models must not overwrite each other.
_ACTIVE_CHAT: contextvars.ContextVar[str] = contextvars.ContextVar("chat_model", default="")
_ACTIVE_EMBED: contextvars.ContextVar[str] = contextvars.ContextVar("embed_model", default="")
_DESCRIBED: dict[str, dict] = {}


def select(chat: str = "", embed: str = "") -> None:
    """Choose the models for this run; "" means fall back to the configured default.

    Always assigns, so a previous run's choice cannot linger in a long-lived
    process (the CLI, or a reused thread).
    """
    _ACTIVE_CHAT.set(chat or "")
    _ACTIVE_EMBED.set(embed or "")


def chat_model() -> str:
    return _ACTIVE_CHAT.get() or OLLAMA_MODEL


def embed_model() -> str:
    return _ACTIVE_EMBED.get() or OLLAMA_EMBED_MODEL


def describe(name: str) -> dict:
    """What a model can do and how big its window is, from Ollama. Cached."""
    if name in _DESCRIBED:
        return _DESCRIBED[name]
    try:
        data = _post("/api/show", {"model": name}, timeout=20)
    except LLMError:
        return {}                       # not cached, so it is retried when Ollama is back
    context = next((v for k, v in (data.get("model_info") or {}).items()
                    if k.endswith(".context_length") and isinstance(v, int)), None)
    info = {
        "capabilities": [str(c) for c in (data.get("capabilities") or [])],
        "context": context,
        "parameters": (data.get("details") or {}).get("parameter_size", ""),
    }
    _DESCRIBED[name] = info
    return info


def num_ctx_for(name: str) -> int:
    """Never ask a model for more context than it has, whatever OLLAMA_NUM_CTX says."""
    limit = describe(name).get("context")
    return min(OLLAMA_NUM_CTX, limit) if limit else OLLAMA_NUM_CTX


def installed() -> list[dict]:
    """Every model on this machine, newest first, ready for a picker."""
    try:
        with urllib.request.urlopen(urllib.request.Request(OLLAMA_URL + "/api/tags"),
                                    timeout=5) as resp:
            tags = json.loads(resp.read()).get("models", [])
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return []

    out = []
    for tag in tags:
        name = tag.get("name", "")
        info = describe(name)
        caps = info.get("capabilities") or []
        out.append({
            "name": name,
            "size_gb": round(tag.get("size", 0) / 1e9, 1),
            "parameters": info.get("parameters") or (tag.get("details") or {}).get("parameter_size", ""),
            "context": info.get("context"),
            # Ollama reports what the model is for, so no guessing from its name.
            "can_chat": "completion" in caps or not caps,
            "can_embed": "embedding" in caps,
        })
    return out


def _post(path: str, payload: dict, timeout: float = OLLAMA_TIMEOUT) -> dict:
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise LLMError(f"Ollama {path} returned {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Cannot reach Ollama at {OLLAMA_URL}: {e}") from e


def max_chars(num_ctx: int, system: str = "", reply_tokens: int = REPLY_TOKENS) -> int:
    """How many characters of user content fit in this context window."""
    free = num_ctx - reply_tokens - TEMPLATE_TOKENS - math.ceil(len(system) / CHARS_PER_TOKEN)
    return max(400, int(free * CHARS_PER_TOKEN))


def max_user_chars(system: str = "", reply_tokens: int = REPLY_TOKENS) -> int:
    """The budget for the model this run selected, capped to its own window."""
    return max_chars(num_ctx_for(chat_model()), system, reply_tokens)


def truncate(text: str, limit: int | None = None) -> str:
    """Keep the head and the tail, never exceeding `limit` characters in total."""
    limit = max_user_chars() if limit is None else limit
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATED_MARK):      # no room to say it was cut
        return text[:max(0, limit)]
    keep = limit - len(TRUNCATED_MARK)
    head = keep // 2
    tail = keep - head
    return text[:head] + TRUNCATED_MARK + (text[-tail:] if tail else "")


def pack(system: str, sections: list[tuple[str, str]],
         reply_tokens: int = REPLY_TOKENS) -> tuple[str, list[str]]:
    """Join labelled sections into one user message that fits the context window.

    Short sections are kept whole and the budget they do not use is shared out among
    the long ones, so a huge submission can never crowd out the assignment text or
    the question list. Returns (message, labels that had to be trimmed).
    """
    budget = max_user_chars(system, reply_tokens)
    budget = max(200, budget - sum(len(label) + 20 for label, _ in sections))
    bodies = [body or "" for _, body in sections]

    allowed = [0] * len(bodies)
    pending, left = list(range(len(bodies))), budget
    while pending:
        share = max(1, left // len(pending))
        fits = [i for i in pending if len(bodies[i]) <= share]
        if not fits:                                   # every section still wants more
            for i in pending:
                allowed[i] = share
            break
        for i in fits:
            allowed[i] = len(bodies[i])
            left -= len(bodies[i])
            pending.remove(i)

    parts, trimmed = [], []
    for (label, _), body, cap in zip(sections, bodies, allowed):
        if len(body) > cap:
            body = truncate(body, cap)
            trimmed.append(label)
        parts.append(f"{label}:\n{body or '(none)'}")
    return "\n\n".join(parts), trimmed


def pack_json(system: str, sections: list[tuple[str, str]], model: str | None = None,
              reply_tokens: int = REPLY_TOKENS) -> tuple[dict, list[str]]:
    """pack() the sections, ask for JSON, and report what had to be trimmed."""
    user, trimmed = pack(system, sections, reply_tokens)
    return chat_json(system, user, model, reply_tokens), trimmed


def status() -> dict:
    """Report whether Ollama is reachable and which selected models are installed."""
    req = urllib.request.Request(OLLAMA_URL + "/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return {"ok": False, "error": str(e), "url": OLLAMA_URL,
                "model": chat_model(), "embed_model": embed_model()}

    def is_installed(name: str) -> bool:
        return any(m == name or m.split(":")[0] == name or m == name + ":latest" for m in models)

    chat, embed = chat_model(), embed_model()
    return {"ok": True, "url": OLLAMA_URL, "models": models,
            "model": chat, "model_ok": is_installed(chat),
            "embed_model": embed, "embed_ok": is_installed(embed),
            "num_ctx": num_ctx_for(chat)}


def chat_json(system: str, user: str, model: str | None = None,
              reply_tokens: int = REPLY_TOKENS) -> dict:
    """Ask the model for a JSON object and parse it.

    The user message is capped to what is left of the context window after the
    system prompt and the reply reserve. num_predict is then set to *all* the
    context the prompt did not use (never less than the reserve), so a short
    prompt still allows a long answer and prompt + answer cannot overflow.
    """
    name = model or chat_model()
    num_ctx = num_ctx_for(name)
    user = truncate(user, max_chars(num_ctx, system, reply_tokens))
    used = math.ceil((len(system) + len(user)) / CHARS_PER_TOKEN) + TEMPLATE_TOKENS
    allow = max(reply_tokens, num_ctx - used)

    resp = _post("/api/chat", {
        "model": name,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": allow},
    })
    content = resp.get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except ValueError as e:
        # A reply cut off by the token limit is invalid JSON for a different reason
        # than a model that ignored the format; say which, or the log misleads.
        if resp.get("done_reason") == "length":
            raise LLMError(
                f"{name} ran past {allow} tokens without finishing its answer. Try a "
                f"shorter submission, a larger OLLAMA_NUM_CTX, or a stronger model.") from e
        raise LLMError(f"Model did not return valid JSON: {content[:200]!r}") from e
    if not isinstance(data, dict):
        raise LLMError(f"Model returned {type(data).__name__}, expected an object")
    return data


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    # The embedding model's window is much smaller than the chat model's; over that,
    # Ollama silently drops the start of each text and the vectors stop comparing.
    name = model or embed_model()
    window = describe(name).get("context") or OLLAMA_EMBED_NUM_CTX
    limit = max_chars(min(OLLAMA_EMBED_NUM_CTX, window), reply_tokens=0)
    resp = _post("/api/embed", {"model": name,
                                "input": [truncate(t, limit) for t in texts]})
    vectors = resp.get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise LLMError("Embedding response did not match the input")
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
