"""Minimal client for a local Ollama server (e.g. Qwen models).

Uses only the standard library so no extra dependency is needed.
"""
from __future__ import annotations

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

# Keep prompts well inside the context window.
MAX_PROMPT_CHARS = int(os.environ.get("OLLAMA_MAX_CHARS", "12000"))


class LLMError(RuntimeError):
    pass


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


def truncate(text: str, limit: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def status() -> dict:
    """Report whether Ollama is reachable and which configured models are installed."""
    req = urllib.request.Request(OLLAMA_URL + "/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return {"ok": False, "error": str(e), "url": OLLAMA_URL,
                "model": OLLAMA_MODEL, "embed_model": OLLAMA_EMBED_MODEL}

    def installed(name: str) -> bool:
        return any(m == name or m.split(":")[0] == name or m == name + ":latest" for m in models)

    return {"ok": True, "url": OLLAMA_URL, "models": models,
            "model": OLLAMA_MODEL, "model_ok": installed(OLLAMA_MODEL),
            "embed_model": OLLAMA_EMBED_MODEL, "embed_ok": installed(OLLAMA_EMBED_MODEL)}


def chat_json(system: str, user: str, model: str | None = None) -> dict:
    """Ask the model for a JSON object and parse it."""
    resp = _post("/api/chat", {
        "model": model or OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": truncate(user)}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
    })
    content = resp.get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except ValueError as e:
        raise LLMError(f"Model did not return valid JSON: {content[:200]!r}") from e
    if not isinstance(data, dict):
        raise LLMError(f"Model returned {type(data).__name__}, expected an object")
    return data


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    resp = _post("/api/embed", {"model": model or OLLAMA_EMBED_MODEL,
                                "input": [truncate(t, 8000) for t in texts]})
    vectors = resp.get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise LLMError("Embedding response did not match the input")
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
