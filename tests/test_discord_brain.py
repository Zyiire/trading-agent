"""Stan's brain: the Ollama tool loop, against a scripted fake Ollama."""
from __future__ import annotations

import json

import httpx
import pytest

from agent.discord_bot import brain as B
from agent.discord_bot import tools as T

URL = "http://127.0.0.1:11434"


def reply_msg(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return httpx.Response(200, json={"message": msg}, request=httpx.Request("POST", URL))


def call(name, args=None):
    return {"function": {"name": name, "arguments": args if args is not None else {}}}


class FakeOllama:
    def __init__(self, *responses):
        self.responses, self.payloads = list(responses), []

    def __call__(self, url, json=None, timeout=None):
        self.payloads.append(json)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def tools_seen(monkeypatch):
    seen = []

    def fake_run(ctx, name, args):
        seen.append((name, args))
        return {"facts": [f"{name} fact"], "data": {}}

    monkeypatch.setattr(B, "run_tool", fake_run)
    return seen


def test_runs_tools_then_returns_the_answer(tools_seen):
    fake = FakeOllama(reply_msg(tool_calls=[call("account")]), reply_msg("Cash is $61,250.10."))
    brain = B.Brain(T.ToolContext(), post=fake)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hi Zy."}]
    assert brain.reply(history, "How much cash?") == "Cash is $61,250.10."
    assert tools_seen == [("account", {})]
    first = fake.payloads[0]
    assert first["model"] == "qwen3:14b" and first["stream"] is False and first["think"] is False
    assert first["options"]["num_ctx"] == 8192 and len(first["tools"]) == 10
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant", "user"]
    tool_msg = fake.payloads[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_name"] == "account"
    assert json.loads(tool_msg["content"]) == {"facts": ["account fact"]}   # no `data`, to save context


def test_string_arguments_and_think_tags(tools_seen):
    fake = FakeOllama(reply_msg(tool_calls=[call("trading_log", '{"limit": 2}')]),
                      reply_msg("<think>hmm</think>\nNo trade was needed."))
    assert B.Brain(T.ToolContext(), post=fake).reply([], "why no trade?") == "No trade was needed."
    assert tools_seen == [("trading_log", {"limit": 2})]


def test_stops_after_eight_tool_calls(tools_seen):
    loops = [reply_msg(tool_calls=[call("account")]) for _ in range(8)]
    fake = FakeOllama(*loops, reply_msg("Here is what I found."))
    assert B.Brain(T.ToolContext(), post=fake).reply([], "?") == "Here is what I found."
    assert len(tools_seen) == 8 and "tools" not in fake.payloads[-1]


def test_empty_answer(tools_seen):
    assert B.Brain(T.ToolContext(), post=FakeOllama(reply_msg(""))).reply([], "?") == B.EMPTY


def test_ollama_not_running():
    fake = FakeOllama(httpx.ConnectError("refused"))
    assert B.Brain(T.ToolContext(), post=fake).reply([], "?") == B.NOT_RUNNING


def test_model_not_downloaded():
    missing = httpx.Response(404, json={"error": "model 'qwen3:14b' not found"}, request=httpx.Request("POST", URL))
    out = B.Brain(T.ToolContext(), post=FakeOllama(missing)).reply([], "?")
    assert out == "My brain model isn't downloaded yet. On the PC, run: ollama pull qwen3:14b"


def test_timeout():
    fake = FakeOllama(httpx.ReadTimeout("slow"))
    assert B.Brain(T.ToolContext(), post=fake).reply([], "?") == "My brain took too long to answer. Try again in a minute."


def test_system_prompt_has_persona_and_rules():
    p = B.system_prompt()
    assert "You are Stan" in p and "Never do arithmetic." in p and "stan.bat" in p


def test_non_dict_arguments_do_not_break_the_loop():
    fake = FakeOllama(reply_msg(tool_calls=[call("trading_log", "[1]")]), reply_msg("Done."))
    assert B.Brain(T.ToolContext(), post=fake).reply([], "?") == "Done."
    assert "tool" in [m["role"] for m in fake.payloads[1]["messages"]]


def test_tool_error_is_sent_but_data_is_not(monkeypatch):
    monkeypatch.setattr(B, "run_tool", lambda ctx, name, args: {"error": "down", "facts": [], "data": {"big": [1] * 99}})
    fake = FakeOllama(reply_msg(tool_calls=[call("account")]), reply_msg("Can't tell."))
    B.Brain(T.ToolContext(), post=fake).reply([], "?")
    assert json.loads(fake.payloads[1]["messages"][-1]["content"]) == {"error": "down", "facts": []}


@pytest.mark.parametrize("body", [{"nope": 1}, "not json"])
def test_malformed_ok_body_gets_the_brain_problem_reply(body):
    r = httpx.Response(200, json=body, request=httpx.Request("POST", URL)) if isinstance(body, dict) else         httpx.Response(200, content=b"<html>", request=httpx.Request("POST", URL))
    assert B.Brain(T.ToolContext(), post=FakeOllama(r)).reply([], "?") == B.BAD_BODY


def test_fixed_replies_are_recognised():
    assert all(B.is_fixed_reply(t) for t in (B.NOT_RUNNING, B.EMPTY, B.BAD_BODY, B.TIMEOUT,
               "My brain model isn't downloaded yet. On the PC, run: ollama pull x",
               "My brain had a problem (HTTP 500). Try again in a minute."))
    assert not B.is_fixed_reply("Cash is $1.")


def test_main_prints_config_errors_not_a_traceback(monkeypatch, capsys):
    from agent.config import ConfigError

    def boom(**kw):
        raise ConfigError("STAN_JOURNAL is bad")
    monkeypatch.setattr("agent.discord_bot.config.load_chat_settings", boom)
    assert B.main(["hi"]) == 2
    assert "STAN_JOURNAL is bad" in capsys.readouterr().out
