"""Stan's brain for Discord: a free local model in Ollama that answers from read-only tools.

Run one question from a terminal (needs Ollama, not Discord):
    python -m agent.discord_bot.brain "how did today go?"
"""
from __future__ import annotations

import json
import re
import sys

import httpx

from agent.config import ConfigError
from agent.discord_bot.tools import ToolContext, run_tool, tool_specs
from agent.stan.persona import STAN

MAX_TOOL_CALLS = 8
NUM_CTX = 8192
NOT_RUNNING = "My brain isn't running. Start Ollama on the PC and ask again."
EMPTY = "I couldn't finish that answer. Try asking another way."
BAD_BODY = "My brain sent back something I couldn't read. Try again in a minute."
TIMEOUT = "My brain took too long to answer. Try again in a minute."
MODEL_MISSING_PREFIX = "My brain model isn't downloaded yet."
HTTP_PROBLEM_PREFIX = "My brain had a problem (HTTP"


def is_fixed_reply(text: str) -> bool:
    """True for the canned problem lines, which shouldn't be remembered as things Stan said."""
    return text in (NOT_RUNNING, EMPTY, BAD_BODY, TIMEOUT) or text.startswith((MODEL_MISSING_PREFIX, HTTP_PROBLEM_PREFIX))

RULES = """How you answer Zy in Discord:
- Always call the tools that answer the question before replying. Never say you will check later. Any question about money, holdings, trades, school results, the schedule or readiness for real money needs at least one tool call.
- Only state numbers and facts that a tool returned in this conversation, word for word from its "facts" where you can. Never do arithmetic.
- If a tool returns an "error", say which information you couldn't get. If no tool covers the question, say you don't know.
- Answer in one to four plain sentences, in the first person. Use a short list only for several items.
- Be honest when results are bad. Don't make bad results sound good.
- You can't trade, pause, change the risk tier or start a study term from Discord. If asked, say so and point Zy to the web app (double-click stan.bat) or the automatic schedule.
- You trade paper money only. Never suggest that real money is being traded."""


def system_prompt() -> str:
    principles = "\n".join(f"- {p}" for p in STAN["principles"])
    return (f"You are {STAN['name']}, a {STAN['title'].lower()}. Your goal: {STAN['goal']}\n{STAN['bio']}\n"
            f"Your principles:\n{principles}\n\n{RULES}")


class ModelMissing(Exception):
    pass


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class Brain:
    def __init__(self, ctx: ToolContext, model: str = "qwen3:14b", url: str = "http://127.0.0.1:11434",
                 thinking: bool = False, post=None, timeout: float = 300.0) -> None:
        self.ctx, self.model, self.url, self.thinking, self.timeout = ctx, model, url.rstrip("/"), thinking, timeout
        self._post = post or httpx.post

    def _chat(self, messages: list[dict], with_tools: bool) -> dict:
        payload = {"model": self.model, "messages": list(messages), "stream": False, "think": self.thinking,
                   "options": {"num_ctx": NUM_CTX}}
        if with_tools:
            payload["tools"] = tool_specs()
        r = self._post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        if r.status_code == 404 and "not found" in r.text.lower():
            raise ModelMissing()
        r.raise_for_status()
        return r.json()["message"]

    def reply(self, history: list[dict], message: str) -> str:
        messages = [{"role": "system", "content": system_prompt()}, *history, {"role": "user", "content": message}]
        calls = 0
        try:
            for _ in range(MAX_TOOL_CALLS + 1):
                m = self._chat(messages, with_tools=calls < MAX_TOOL_CALLS)
                tool_calls = m.get("tool_calls") or []
                if not tool_calls or calls >= MAX_TOOL_CALLS:
                    return _clean(m.get("content", "")) or EMPTY
                messages.append({"role": "assistant", "content": m.get("content", ""), "tool_calls": tool_calls})
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name, args = fn.get("name", ""), fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    calls += 1
                    out = run_tool(self.ctx, name, args) if calls <= MAX_TOOL_CALLS else \
                        {"error": "Too many lookups for one answer. Answer with what you have."}
                    slim = {k: out[k] for k in ("facts", "error") if k in out}   # keep `data` out of the small context window
                    messages.append({"role": "tool", "tool_name": name, "content": json.dumps(slim)})
            return EMPTY
        except ModelMissing:
            return f"{MODEL_MISSING_PREFIX} On the PC, run: ollama pull {self.model}"
        except httpx.TimeoutException:
            return TIMEOUT
        except httpx.TransportError:
            return NOT_RUNNING
        except (ValueError, KeyError):
            return BAD_BODY
        except httpx.HTTPStatusError as err:
            return f"{HTTP_PROBLEM_PREFIX} {err.response.status_code}). Try again in a minute."


def main(argv: list[str] | None = None) -> int:
    from agent.discord_bot.config import load_chat_settings

    question = " ".join(argv if argv is not None else sys.argv[1:]).strip()
    if not question:
        print('Usage: python -m agent.discord_bot.brain "your question"')
        return 2
    try:
        s = load_chat_settings(require_discord=False)
    except ConfigError as err:
        print(err)
        return 2
    print(Brain(ToolContext(journal_path=s.journal_path), s.model, s.ollama_url, s.thinking).reply([], question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
