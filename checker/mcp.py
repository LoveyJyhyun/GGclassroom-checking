"""A minimal MCP server (Model Context Protocol) for the agent tools.

Speaks the Streamable HTTP transport in its simplest form: every POST carries one
JSON-RPC message (or a batch) and gets a JSON reply. Tool calls need nothing more:
the server never starts a conversation itself, so the optional GET stream is
refused with 405 as the spec allows, and no session state is kept. Standard
library only, like the rest of the checker.

Clients known to work: Dify's MCP tools (URL must end in /mcp, which is how Dify
picks this transport), the official MCP Python SDK, Claude Code.
"""
from __future__ import annotations

from . import agent_tools

SERVER_NAME = "ggclassroom-checker"
SERVER_VERSION = "1.1.0"
LATEST = "2025-06-18"
SUPPORTED = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
INSTRUCTIONS = (
    "Tools for triaging Google Classroom check reports. Students appear only as codes "
    "(S01, S02 ...). Copying and AI-writing signals are leads for the teacher, never proof: "
    "never call a student a cheater. Start with triage_brief, use student_detail to dig in, "
    "and finish with save_triage. Grades are saved to Classroom only by the teacher."
)

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602


def error_response(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _result(msg_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def tool_list() -> list[dict]:
    return [{
        "name": t.name, "title": t.title, "description": t.description,
        "inputSchema": t.schema(),
        "annotations": {"title": t.title, "readOnlyHint": t.read_only,
                        "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    } for t in agent_tools.TOOLS.values()]


def _call_tool(msg_id, params: dict) -> dict:
    name = params.get("name")
    if name not in agent_tools.TOOLS:
        return error_response(msg_id, INVALID_PARAMS, f"Unknown tool: {name}")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return error_response(msg_id, INVALID_PARAMS, "arguments must be an object")
    try:
        text, data = agent_tools.call(name, arguments)
    except agent_tools.ToolError as e:
        # A tool-level failure the model can read and correct, not a protocol error.
        return _result(msg_id, {"content": [{"type": "text", "text": str(e)}], "isError": True})
    except (OSError, ValueError, TypeError) as e:
        return _result(msg_id, {"content": [{"type": "text", "text": f"Tool failed: {e}"}],
                                "isError": True})
    return _result(msg_id, {"content": [{"type": "text", "text": text}],
                            "structuredContent": data, "isError": False})


def handle(message) -> dict | None:
    """Answer one JSON-RPC message. Returns None for notifications (no reply)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
        return error_response(message.get("id") if isinstance(message, dict) else None,
                      INVALID_REQUEST, "Not a JSON-RPC 2.0 request")
    if "id" not in message:                  # notifications/initialized, cancelled, ...
        return None
    msg_id, method = message["id"], message["method"]
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return error_response(msg_id, INVALID_PARAMS, "params must be an object")

    if method == "initialize":
        asked = params.get("protocolVersion")
        return _result(msg_id, {
            "protocolVersion": asked if asked in SUPPORTED else LATEST,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "title": "GGClassroom Checker", "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": tool_list()})
    if method == "tools/call":
        return _call_tool(msg_id, params)
    if method in ("resources/list", "prompts/list"):
        return _result(msg_id, {method.split("/")[0]: []})
    return error_response(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")


def handle_body(body) -> list[dict] | dict | None:
    """A POST body may be one message or a batch; notifications get no reply."""
    if isinstance(body, list):
        replies = [r for r in (handle(m) for m in body) if r is not None]
        return replies or None
    return handle(body)
