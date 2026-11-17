"""Stan's Discord settings, owner check, message handling and missed-run timing."""
from __future__ import annotations

import pytest

from agent.config import ConfigError
from agent.discord_bot.config import ChatSettings, load_chat_settings

KEYS = ("DISCORD_BOT_TOKEN", "DISCORD_OWNER_ID", "DISCORD_CHANNEL_ID", "STAN_MODEL", "OLLAMA_URL", "STAN_THINK",
        "STAN_JOURNAL")


@pytest.fixture
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_settings_defaults_and_values(clean_env):
    clean_env.setenv("DISCORD_BOT_TOKEN", "tok")
    clean_env.setenv("DISCORD_OWNER_ID", "111")
    clean_env.setenv("DISCORD_CHANNEL_ID", "222")
    s = load_chat_settings(env_file=None)
    assert (s.owner_id, s.channel_id, s.model, s.ollama_url, s.thinking) == (111, 222, "qwen3:14b", "http://127.0.0.1:11434", False)
    assert s.journal_path == "data/stan.db" and s.discord_ready
    assert "tok" not in repr(s)


def test_settings_overrides(clean_env):
    for k, v in {"DISCORD_BOT_TOKEN": "t", "DISCORD_OWNER_ID": "1", "DISCORD_CHANNEL_ID": "2", "STAN_MODEL": "qwen3:8b",
                 "OLLAMA_URL": "http://x:1", "STAN_THINK": "true"}.items():
        clean_env.setenv(k, v)
    s = load_chat_settings(env_file=None)
    assert (s.model, s.ollama_url, s.thinking) == ("qwen3:8b", "http://x:1", True)


def test_missing_discord_settings_are_named(clean_env):
    clean_env.setenv("DISCORD_OWNER_ID", "1")
    with pytest.raises(ConfigError) as err:
        load_chat_settings(env_file=None)
    assert "DISCORD_BOT_TOKEN" in str(err.value) and "DISCORD_CHANNEL_ID" in str(err.value)
    assert "DISCORD_OWNER_ID" not in str(err.value)


def test_missing_discord_settings_allowed_for_the_brain_alone(clean_env):
    s = load_chat_settings(env_file=None, require_discord=False)
    assert not s.discord_ready and s.model == "qwen3:14b"


def test_ids_must_be_numbers(clean_env):
    clean_env.setenv("DISCORD_BOT_TOKEN", "t")
    clean_env.setenv("DISCORD_OWNER_ID", "me")
    clean_env.setenv("DISCORD_CHANNEL_ID", "2")
    with pytest.raises(ConfigError, match="DISCORD_OWNER_ID must be a number"):
        load_chat_settings(env_file=None)


import asyncio
from collections import deque
from datetime import date, datetime

from agent.discord_bot import bot as BOT
from agent.discord_bot.tools import ET

S = ChatSettings(bot_token="t", owner_id=111, channel_id=222)


def test_should_answer_only_the_owner_in_the_channel():
    assert BOT.should_answer(111, False, 222, S)
    assert not BOT.should_answer(999, False, 222, S)      # someone else
    assert not BOT.should_answer(111, True, 222, S)       # a bot
    assert not BOT.should_answer(111, False, 333, S)      # another channel


def at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_due_checks_times_weekends_and_once_a_day():
    done: set = set()
    assert BOT.due_checks(at(2026, 11, 17, 9, 49), done) == []
    assert BOT.due_checks(at(2026, 11, 17, 9, 50), done) == ["open"]
    assert BOT.due_checks(at(2026, 11, 17, 18, 30), done) == ["open", "study"]
    done |= {("open", date(2026, 11, 17)), ("study", date(2026, 11, 17))}
    assert BOT.due_checks(at(2026, 11, 17, 20, 0), done) == []
    assert BOT.due_checks(at(2026, 11, 21, 10, 0), set()) == []   # Saturday


class FakeTyping:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        self.log.append("typing")

    async def __aexit__(self, *exc):
        return False


class FakeChannel:
    def __init__(self):
        self.log, self.sent = [], []

    def typing(self):
        return FakeTyping(self.log)

    async def send(self, text, **kwargs):
        self.sent.append(text)


class FakeBrain:
    def __init__(self, text):
        self.text, self.seen = text, []

    def reply(self, history, message):
        self.seen.append((list(history), message))
        return self.text


def test_handle_message_types_replies_splits_and_remembers():
    channel, brain, history = FakeChannel(), FakeBrain("x" * 2500), deque(maxlen=20)
    asyncio.run(BOT.handle_message("how much cash?", channel, brain, history))
    assert channel.log == ["typing"] and [len(t) for t in channel.sent] == [2000, 500]
    assert list(history) == [{"role": "user", "content": "how much cash?"}, {"role": "assistant", "content": "x" * 2500}]
    asyncio.run(BOT.handle_message("and today?", channel, brain, history))
    assert brain.seen[1][0][0]["content"] == "how much cash?"


def test_history_keeps_only_twenty_messages():
    channel, brain, history = FakeChannel(), FakeBrain("ok"), deque(maxlen=20)
    for i in range(15):
        asyncio.run(BOT.handle_message(f"q{i}", channel, brain, history))
    assert len(history) == 20 and history[0]["content"] == "q5"


def test_main_stops_on_missing_settings(monkeypatch, clean_env, capsys):
    monkeypatch.setattr("agent.discord_bot.config.load_dotenv", None)
    assert BOT.main() == 2
    assert "Missing DISCORD_BOT_TOKEN" in capsys.readouterr().out
