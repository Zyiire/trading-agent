"""Settings for Stan's Discord chat, read from the git-ignored .env file.

The bot token is never printed: it is excluded from repr like the Alpaca keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import ConfigError, _as_bool

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


@dataclass(frozen=True)
class ChatSettings:
    bot_token: str = field(default="", repr=False)
    owner_id: int = 0
    channel_id: int = 0
    model: str = "qwen3:14b"
    ollama_url: str = "http://127.0.0.1:11434"
    thinking: bool = False
    journal_path: str = "data/stan.db"

    @property
    def discord_ready(self) -> bool:
        return bool(self.bot_token and self.owner_id and self.channel_id)


def _discord_id(name: str) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number (a Discord ID). Copy it with Developer Mode on.") from None


def load_chat_settings(env_file: str | os.PathLike | None = ".env", require_discord: bool = True) -> ChatSettings:
    if load_dotenv is not None and env_file and Path(env_file).exists():
        load_dotenv(env_file)
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    owner, channel = _discord_id("DISCORD_OWNER_ID"), _discord_id("DISCORD_CHANNEL_ID")
    missing = [n for n, v in (("DISCORD_BOT_TOKEN", token), ("DISCORD_OWNER_ID", owner),
                              ("DISCORD_CHANNEL_ID", channel)) if not v]
    if require_discord and missing:
        raise ConfigError("Missing " + ", ".join(missing) + " in .env. See the README section 'Talk to Stan on Discord'.")
    return ChatSettings(
        bot_token=token, owner_id=owner, channel_id=channel,
        model=os.getenv("STAN_MODEL", "").strip() or "qwen3:14b",
        ollama_url=os.getenv("OLLAMA_URL", "").strip() or "http://127.0.0.1:11434",
        thinking=_as_bool(os.getenv("STAN_THINK"), default=False),
        journal_path=os.getenv("STAN_JOURNAL", "").strip() or "data/stan.db",
    )
