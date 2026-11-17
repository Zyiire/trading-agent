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
