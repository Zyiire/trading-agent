"""Post one job report to Stan's Discord channel, then exit. Called at the end of each .bat job:

    python -m agent.discord_bot.notify open 0

It always exits 0, so a report problem can never make the job itself look failed.
"""
from __future__ import annotations

import argparse
import sys

import httpx

from agent.discord_bot.config import load_chat_settings
from agent.discord_bot.reports import JOB_NAMES, job_report, split_message
from agent.discord_bot.tools import ToolContext

API = "https://discord.com/api/v10"
USER_AGENT = "DiscordBot (https://github.com/Zyiire/trading-agent, 1.0)"


def send(token: str, channel_id: int, text: str, post=httpx.post) -> int:
    sent = 0
    for chunk in split_message(text):
        r = post(f"{API}/channels/{channel_id}/messages",
                 headers={"Authorization": f"Bot {token}", "User-Agent": USER_AGENT},
                 json={"content": chunk, "allowed_mentions": {"parse": []}}, timeout=15)
        r.raise_for_status()
        sent += 1
    return sent


def main(argv: list[str] | None = None, ctx: ToolContext | None = None, post=None) -> int:
    parser = argparse.ArgumentParser(description="Send a job report to Stan's Discord channel.")
    parser.add_argument("job", choices=sorted(JOB_NAMES))
    parser.add_argument("code", type=int)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        print("Couldn't send the Discord report: bad arguments (usage: notify open|study|collect EXITCODE).")
        return 0
    try:
        settings = load_chat_settings(require_discord=False)
        if not settings.discord_ready:
            print("Discord not set up, skipping report.")
            return 0
        text = job_report(ctx or ToolContext(journal_path=settings.journal_path), args.job, args.code)
        if not text:
            print("Nothing to report.")
            return 0
        n = send(settings.bot_token, settings.channel_id, text, post=post or httpx.post)
        print(f"Sent the {args.job} report to Discord ({n} message{'s' if n != 1 else ''}).")
    except Exception as err:
        print(f"Couldn't send the Discord report: {type(err).__name__}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
