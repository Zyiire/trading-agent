"""Stan's always-on Discord bot. Answers only the owner, only in Stan's channel.

Run:  python -m agent.discord_bot.bot     (run_discord.bat keeps it running)
Exit codes: 0 closed, 1 crashed (restart), 2 settings, token or intent problem (don't restart).
"""
from __future__ import annotations

import asyncio
import sys
from collections import deque
from datetime import date, datetime, time
from pathlib import Path

import discord
from discord.ext import tasks

from agent.config import ConfigError
from agent.discord_bot.brain import EMPTY, Brain
from agent.discord_bot.config import ChatSettings, load_chat_settings
from agent.discord_bot.reports import missed_run_message, split_message
from agent.discord_bot.tools import ET, ToolContext

CHECKS = (("open", time(9, 50)), ("study", time(18, 30)))
NO_PINGS = discord.AllowedMentions.none()


def should_answer(author_id: int, author_is_bot: bool, channel_id: int, settings: ChatSettings) -> bool:
    return (not author_is_bot) and author_id == settings.owner_id and channel_id == settings.channel_id


def due_checks(now: datetime, done: set[tuple[str, date]]) -> list[str]:
    """Which missed-run checks are due now: weekdays, after their time, once per day."""
    if now.weekday() >= 5:
        return []
    return [job for job, at in CHECKS if now.time() >= at and (job, now.date()) not in done]


async def handle_message(content: str, channel, brain, history: deque) -> None:
    async with channel.typing():
        text = await asyncio.to_thread(brain.reply, list(history), content)
    history.append({"role": "user", "content": content})
    history.append({"role": "assistant", "content": text})
    for chunk in split_message(text) or [EMPTY]:
        await channel.send(chunk, allowed_mentions=NO_PINGS)


class StanBot(discord.Client):
    def __init__(self, settings: ChatSettings, brain: Brain, data_dir: Path = Path("data")) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, allowed_mentions=NO_PINGS)
        self.settings, self.brain, self.data_dir = settings, brain, Path(data_dir)
        self.history: deque = deque(maxlen=20)
        self.done: set[tuple[str, date]] = set()
        self.lock = asyncio.Lock()

    async def setup_hook(self) -> None:
        self.missed_run_loop.start()

    async def on_ready(self) -> None:
        print(f"Stan is online as {self.user}, listening in channel {self.settings.channel_id}.", flush=True)

    async def on_message(self, message: discord.Message) -> None:
        if not should_answer(message.author.id, message.author.bot, message.channel.id, self.settings):
            return
        if not message.content.strip():
            return
        async with self.lock:
            await handle_message(message.content, message.channel, self.brain, self.history)

    async def _channel(self):
        return self.get_channel(self.settings.channel_id) or await self.fetch_channel(self.settings.channel_id)

    @tasks.loop(minutes=1)
    async def missed_run_loop(self) -> None:
        now = datetime.now(ET)
        for job in due_checks(now, self.done):
            self.done.add((job, now.date()))
            text = missed_run_message(job, self.data_dir, now.date())
            if text:
                await (await self._channel()).send(text)

    @missed_run_loop.before_loop
    async def _before_missed_run_loop(self) -> None:
        await self.wait_until_ready()


def main() -> int:
    try:
        settings = load_chat_settings()
    except ConfigError as err:
        print(err, flush=True)
        return 2
    brain = Brain(ToolContext(journal_path=settings.journal_path), settings.model, settings.ollama_url, settings.thinking)
    try:
        StanBot(settings, brain).run(settings.bot_token)
    except discord.LoginFailure:
        print("Discord rejected DISCORD_BOT_TOKEN. Reset the token in the Developer Portal and update .env.", flush=True)
        return 2
    except discord.PrivilegedIntentsRequired:
        print("Turn on Message Content Intent: Discord Developer Portal > Stan > Bot > Message Content Intent.", flush=True)
        return 2
    except Exception as err:
        print(f"Stan's bot stopped: {type(err).__name__}: {err}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
