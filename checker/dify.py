"""Start the Dify triage workflow after a check, and record how it went.

The workflow (examples/dify/ggclassroom-triage.yml) reads the report through this
app's MCP tools and writes its triage back with save_triage, so this module only
has to start it and notice if it finished without saving anything.

Off unless DIFY_TRIAGE_KEY is set (the workflow app's API key, "app-...").
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import triage

DIFY_API_URL = os.environ.get("DIFY_API_URL", "http://localhost/v1").rstrip("/")
DIFY_TRIAGE_KEY = os.environ.get("DIFY_TRIAGE_KEY", "")
DIFY_TIMEOUT = float(os.environ.get("DIFY_TIMEOUT", "900"))
TRIAGE_LANGUAGE = os.environ.get("TRIAGE_LANGUAGE", "Vietnamese")


class DifyError(RuntimeError):
    pass


def configured() -> bool:
    return bool(DIFY_TRIAGE_KEY)


def run_workflow(inputs: dict, key: str | None = None, timeout: float | None = None) -> dict:
    """Run a Dify workflow and wait for it. Returns the workflow_finished data.

    Streaming mode, because a blocking request can be cut off by a proxy long before a
    local model has finished; the stream keeps the connection busy until the end.
    """
    req = urllib.request.Request(
        DIFY_API_URL + "/workflows/run",
        data=json.dumps({"inputs": inputs, "response_mode": "streaming",
                         "user": "ggclassroom-checker"}).encode(),
        headers={"Authorization": f"Bearer {key or DIFY_TRIAGE_KEY}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or DIFY_TIMEOUT) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue                          # blank lines, "event: ping"
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("event") == "workflow_finished":
                    return event.get("data") or {}
                if event.get("event") == "error":
                    raise DifyError(event.get("message") or "Dify reported an error")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise DifyError(f"Dify returned {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise DifyError(f"Cannot reach Dify at {DIFY_API_URL}: {e}") from e
    raise DifyError("Dify closed the stream before the workflow finished")


def run_triage(report_path: str) -> dict:
    """Triage one saved report with the Dify workflow. Never raises: the outcome is stored."""
    triage.write_triage(report_path, state="running", error=None)
    try:
        finished = run_workflow({"report": os.path.basename(report_path),
                                 "language": TRIAGE_LANGUAGE})
    except DifyError as e:
        return triage.write_triage(report_path, state="failed", error=str(e))
    except Exception as e:                        # noqa: BLE001 - e.g. a stream cut mid-chunk
        return triage.write_triage(report_path, state="failed",
                                   error=f"Lost the connection to Dify: {type(e).__name__}: {e}")

    stored = triage.load_triage(report_path) or {}
    if stored.get("state") == "done":                 # save_triage was called by the workflow
        return stored
    status = finished.get("status", "unknown")
    error = finished.get("error") or (
        "the workflow finished without calling save_triage" if status == "succeeded" else "")
    return triage.write_triage(report_path, state="failed",
                               error=f"Dify workflow {status}: {error}".rstrip(": "))
