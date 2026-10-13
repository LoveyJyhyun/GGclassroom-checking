"""Keeping prompts inside the model's context window."""
from checker import llm


def test_budget_follows_num_ctx(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_NUM_CTX", 8192)
    big = llm.max_user_chars()
    monkeypatch.setattr(llm, "OLLAMA_NUM_CTX", 2048)
    small = llm.max_user_chars()
    assert small < big                      # lowering num_ctx really shrinks the prompt

    # The reply reserve and the system prompt both come out of the same budget.
    assert llm.max_user_chars("x" * 3000) < llm.max_user_chars("")
    assert llm.max_user_chars(reply_tokens=1500) < llm.max_user_chars(reply_tokens=100)


def test_budget_never_goes_negative(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_NUM_CTX", 256)
    assert llm.max_user_chars("x" * 100_000, reply_tokens=4000) >= 400


def test_truncate_respects_the_limit_exactly():
    # Tiny limits have no room for the "[truncated]" marker and must still not overflow.
    for limit in (0, 1, 5, 15, 19):
        assert len(llm.truncate("a" * 900, limit)) <= limit

    for limit in (400, 1000, 7777):
        out = llm.truncate("a" * 50_000, limit)
        assert len(out) == limit
        assert llm.TRUNCATED_MARK in out
        assert out.startswith("a") and out.endswith("a")
    assert llm.truncate("short", 400) == "short"


def test_pack_keeps_short_sections_whole_and_shares_the_rest():
    message, trimmed = llm.pack("sys", [
        ("Assignment", "Add two numbers."),
        ("Submission A", "x" * 40_000),
        ("Submission B", "y" * 40_000),
    ])
    assert "Add two numbers." in message          # the brief is never the thing cut
    assert trimmed == ["Submission A", "Submission B"]
    assert abs(message.count("x") - message.count("y")) <= 2   # shared evenly
    assert len(message) <= llm.max_user_chars("sys")


def test_pack_reports_nothing_when_everything_fits():
    message, trimmed = llm.pack("sys", [("Code", "print(1)"), ("Tests", "1 -> 1")])
    assert trimmed == []
    assert "print(1)" in message and "1 -> 1" in message


def test_pack_gives_a_lone_long_section_the_whole_budget():
    one, _ = llm.pack("sys", [("Student answer", "z" * 100_000)])
    two, _ = llm.pack("sys", [("Student answer", "z" * 100_000), ("Other", "w" * 100_000)])
    assert one.count("z") > two.count("z")


def test_embedding_budget_is_smaller_than_the_chat_budget():
    """nomic-embed-text has a 2048-token window; over it, Ollama drops the start."""
    assert llm.max_chars(llm.OLLAMA_EMBED_NUM_CTX, reply_tokens=0) < llm.max_user_chars()


# --- choosing a model -------------------------------------------------------

def test_select_falls_back_and_does_not_linger(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_MODEL", "default-chat")
    monkeypatch.setattr(llm, "OLLAMA_EMBED_MODEL", "default-embed")

    llm.select("qwen3:14b", "bge-m3")
    assert llm.chat_model() == "qwen3:14b" and llm.embed_model() == "bge-m3"

    llm.select()                       # a later run with no choice gets the default back
    assert llm.chat_model() == "default-chat"
    assert llm.embed_model() == "default-embed"


def test_a_selection_is_per_thread(monkeypatch):
    """The web app runs each check in a thread; two runs must not collide."""
    import threading
    monkeypatch.setattr(llm, "OLLAMA_MODEL", "default-chat")
    llm.select("picked-by-main")
    seen = {}

    def worker():
        seen["before"] = llm.chat_model()
        llm.select("picked-by-thread")
        seen["after"] = llm.chat_model()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["before"] == "default-chat"        # does not inherit the main choice
    assert seen["after"] == "picked-by-thread"
    assert llm.chat_model() == "picked-by-main"    # and cannot overwrite it
    llm.select()


def test_num_ctx_is_capped_to_the_models_own_window(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_NUM_CTX", 8192)
    monkeypatch.setattr(llm, "_DESCRIBED", {
        "small": {"context": 2048, "capabilities": ["completion"]},
        "big": {"context": 262144, "capabilities": ["completion"]},
        "unknown": {},
    })
    assert llm.num_ctx_for("small") == 2048        # never ask for more than it has
    assert llm.num_ctx_for("big") == 8192          # and never more than configured
    assert llm.num_ctx_for("unknown") == 8192      # no metadata -> the configured value

    llm.select("small")
    assert llm.max_user_chars() < llm.max_chars(8192)
    llm.select()


def test_installed_returns_nothing_when_ollama_is_down(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_URL", "http://127.0.0.1:1")
    assert llm.installed() == []
