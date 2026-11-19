# Stan on Discord Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zy chats with Stan in one Discord channel. A free local model (qwen3:14b in Ollama) answers from read-only tools, and fixed-wording reports arrive after each scheduled job.

**Architecture:** A new package, `agent/discord_bot/`. `tools.py` holds read-only functions that return `{"facts": [...], "data": {...}}`. `brain.py` runs an Ollama tool-calling loop over those tools. `reports.py` builds report text without the model. `notify.py` posts reports through Discord's REST API from the `.bat` jobs. `bot.py` is the always-on `discord.py` client. A PowerShell-registered scheduled task keeps the bot running.

**Tech Stack:** Python 3.14 on Windows 11, `discord.py` 2.6+, `httpx` (already a dependency), Ollama 0.40 at `http://127.0.0.1:11434`, `pytest`, Windows Task Scheduler.

**Spec:** `docs/superpowers/specs/2026-11-16-stan-discord-design.md`

## Global Constraints

- Paper trading only, and nothing in `agent/discord_bot/` may place, cancel or change orders, pause or resume trading, change the tier, or start training.
- Stan answers only messages where author ID == `DISCORD_OWNER_ID`, the channel ID == `DISCORD_CHANNEL_ID`, and the author isn't a bot.
- Default model `qwen3:14b`; `STAN_MODEL`, `OLLAMA_URL` (default `http://127.0.0.1:11434`), `STAN_THINK` (default `false`) can override.
- Ollama requests: `stream: false`, `options.num_ctx` 8192, at most 8 tool calls per reply.
- History: last 20 messages, in memory only.
- Discord messages are at most 2,000 characters each.
- Every tool result has the shape `{"facts": [str, ...], "data": {...}}`, or `{"error": str}` on failure.
- Reports never use the model. `notify` always exits 0, so a report can never make a job look failed.
- The bot token is never printed or logged.
- All tests run offline: no Discord, Ollama or Alpaca access.
- New York time is `agent.stan.zerodte.bars.ET` (`ZoneInfo("America/New_York")`).
- Run tests with `python -m pytest -q` from the repo root `c:\Users\space\Downloads\trading-agent\trading-agent`.

## Review Focus

1. **A day with no orders**, which is common (after hours, or nothing to rebalance). The open report must say "No orders today" plus Stan's own notes, not show an empty message. Covered in Task 6.
2. **Log files that don't exist yet, or are cut off mid-run** (the PC slept during a job, so there's a header but no "Exit code" line). The schedule tool and the missed-run check must treat these as "no entry" or "didn't finish", not crash. Covered in Tasks 3 and 6.
3. **The model replies in a form the loop doesn't expect**: tool `arguments` as a JSON string instead of an object, `<think>` tags in the text, an empty reply, or an unknown tool name. Each must turn into a sensible reply. Covered in Task 5.
4. **A reply longer than 2,000 characters with no line breaks** (for example a long URL or table). The splitter must still cut it under the limit. Covered in Task 6.
5. **The study term is still running** when Zy asks about school (`meta.state == "training"`). Stan must say he's in class and that the numbers are from the last finished term. Covered in Task 4.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `discord.py` |
| `.env.example` | Modify | Document the six new settings |
| `agent/broker/base.py` | Modify | `OrderInfo.submitted_at`; `recent_orders` in the `Broker` protocol |
| `agent/broker/alpaca.py` | Modify | `recent_orders(limit)` read method; fill `submitted_at` |
| `agent/discord_bot/__init__.py` | Create | Package marker and a one-line docstring |
| `agent/discord_bot/config.py` | Create | `ChatSettings`, `load_chat_settings` |
| `agent/discord_bot/tools.py` | Create | Formatting helpers, log parsing, the ten tools, the tool registry |
| `agent/discord_bot/brain.py` | Create | System prompt, Ollama tool loop, terminal command |
| `agent/discord_bot/reports.py` | Create | `split_message`, report text, missed-run message |
| `agent/discord_bot/notify.py` | Create | Discord REST send, `notify` command |
| `agent/discord_bot/bot.py` | Create | Owner check, message handling, missed-run loop, `main` |
| `run_open.bat`, `run_study.bat`, `run_collect.bat` | Modify | Save the exit code, then call `notify` |
| `run_discord.bat` | Create | Start the bot, restart on crash, stop on settings errors |
| `schedule_discord.ps1` | Create | Register the "Stan Discord" logon task with no time limit |
| `schedule_trading.bat` | Modify | Call `schedule_discord.ps1` |
| `README.md` | Modify | Discord setup and use |
| `tests/test_discord_tools.py`, `tests/test_discord_brain.py`, `tests/test_discord_reports.py`, `tests/test_discord_bot.py`, `tests/test_broker_orders.py` | Create | Tests |

---

### Task 1: Dependencies and settings

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Create: `agent/discord_bot/__init__.py`
- Create: `agent/discord_bot/config.py`
- Test: `tests/test_discord_bot.py` (settings tests; later tasks add more to this file)

**Interfaces:**
- Consumes: `agent.config.ConfigError`, `agent.config._as_bool`
- Produces: `ChatSettings` (frozen dataclass: `bot_token: str` (not shown in repr), `owner_id: int`, `channel_id: int`, `model: str`, `ollama_url: str`, `thinking: bool`, `journal_path: str`; property `discord_ready -> bool`); `load_chat_settings(env_file: str | None = ".env", require_discord: bool = True) -> ChatSettings`

- [ ] **Step 1: Install discord.py and confirm it imports on Python 3.14**

Run: `python -m pip install "discord.py>=2.6,<3"`
Then: `python -c "import discord; print(discord.__version__)"`
Expected: prints `2.6.x` or `2.7.x`.

- [ ] **Step 2: Add the dependency**

Append to `requirements.txt`:

```
discord.py>=2.6,<3  # Stan's Discord chat (agent/discord_bot)
```

- [ ] **Step 3: Write the failing settings tests**

Create `tests/test_discord_bot.py`:

```python
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
```

- [ ] **Step 4: Run them to make sure they fail**

Run: `python -m pytest tests/test_discord_bot.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.discord_bot'`.

- [ ] **Step 5: Write the package and settings**

Create `agent/discord_bot/__init__.py`:

```python
"""Stan in Discord: read-only tools, a local-model brain, reports, and the bot."""
```

Create `agent/discord_bot/config.py`:

```python
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
```

- [ ] **Step 6: Run the tests to make sure they pass**

Run: `python -m pytest tests/test_discord_bot.py -q`
Expected: 5 passed.

- [ ] **Step 7: Document the settings**

Append to `.env.example`:

```
# Stan on Discord (see the README section "Talk to Stan on Discord").
# Never commit these or paste them into a chat.
DISCORD_BOT_TOKEN=
DISCORD_OWNER_ID=
DISCORD_CHANNEL_ID=
# Optional: Stan's local brain in Ollama (free, runs on this PC's GPU)
# STAN_MODEL=qwen3:14b
# OLLAMA_URL=http://127.0.0.1:11434
# STAN_THINK=false
```

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .env.example agent/discord_bot/__init__.py agent/discord_bot/config.py tests/test_discord_bot.py
git commit -m "Discord chat: settings and discord.py dependency"
```

---

### Task 2: Broker `recent_orders` (read-only)

**Files:**
- Modify: `agent/broker/base.py` (the `OrderInfo` dataclass and the `Broker` protocol)
- Modify: `agent/broker/alpaca.py` (`_order_info` and a new method after `get_order`)
- Test: `tests/test_broker_orders.py`

**Interfaces:**
- Produces: `OrderInfo.submitted_at: datetime | None = None` (new last field, defaulted, so existing positional constructions still work); `AlpacaBroker.recent_orders(limit: int = 20) -> list[OrderInfo]`, newest first.

- [ ] **Step 1: Write the failing test**

Create `tests/test_broker_orders.py`:

```python
"""The read-only order list the Discord tools use."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from agent.broker.alpaca import AlpacaBroker
from agent.broker.base import OrderInfo
from agent.config import Settings


class FakeTrading:
    def __init__(self):
        self.requests = []

    def get_orders(self, req):
        self.requests.append(req)
        t = datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc)
        return [SimpleNamespace(client_order_id="stan-SPY-1-buy", symbol="SPY", side=SimpleNamespace(value="buy"),
                                qty="52.31", status=SimpleNamespace(value="filled"), filled_qty="52.31",
                                filled_avg_price="663.57", filled_at=t, submitted_at=t)]


def test_recent_orders_lists_all_statuses_newest_first():
    trading = FakeTrading()
    broker = AlpacaBroker(Settings(api_key="k", secret_key="s"), trading, object(), object())
    orders = broker.recent_orders(5)
    assert orders == [OrderInfo("stan-SPY-1-buy", "SPY", "buy", 52.31, "filled", 52.31, 663.57,
                                datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc),
                                datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc))]
    req = trading.requests[0]
    assert req.limit == 5 and str(getattr(req.status, "value", req.status)) == "all"


def test_order_info_still_builds_without_submitted_at():
    o = OrderInfo("id", "SPY", "buy", 1.0, "accepted", 0.0, None, None)
    assert o.submitted_at is None
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_broker_orders.py -q`
Expected: FAIL with `AttributeError: 'AlpacaBroker' object has no attribute 'recent_orders'`, plus a `TypeError` or `AttributeError` on `submitted_at`.

- [ ] **Step 3: Add the field and the protocol method**

In `agent/broker/base.py`, add the last line to `OrderInfo`:

```python
@dataclass(frozen=True)
class OrderInfo:
    client_order_id: str
    symbol: str
    side: str          # "buy" or "sell"
    qty: float
    status: str        # Alpaca status, e.g. "accepted", "filled", "canceled"
    filled_qty: float
    filled_avg_price: float | None
    filled_at: datetime | None
    submitted_at: datetime | None = None
```

In the `Broker` protocol, after `get_order`, add:

```python
    def recent_orders(self, limit: int = 20) -> list[OrderInfo]: ...
```

- [ ] **Step 4: Implement the read method**

In `agent/broker/alpaca.py`, change `_order_info` to also fill `submitted_at`:

```python
    @staticmethod
    def _order_info(o) -> OrderInfo:
        return OrderInfo(
            client_order_id=str(o.client_order_id),
            symbol=o.symbol,
            side=_enum_text(o.side),
            qty=_f(o.qty) or 0.0,
            status=_enum_text(o.status),
            filled_qty=_f(o.filled_qty) or 0.0,
            filled_avg_price=_f(o.filled_avg_price),
            filled_at=o.filled_at,
            submitted_at=getattr(o, "submitted_at", None),
        )
```

Keep whatever decorator `_order_info` already has. If it isn't a `@staticmethod`, keep its current form and only add the `submitted_at=` line. After `get_order`, add:

```python
    def recent_orders(self, limit: int = 20) -> list[OrderInfo]:
        """Read-only: the account's latest orders of any status, newest first."""
        orders = self._trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=int(limit)))
        return [self._order_info(o) for o in orders]
```

- [ ] **Step 5: Run the new tests and the whole suite**

Run: `python -m pytest tests/test_broker_orders.py -q` and then `python -m pytest -q`
Expected: 2 passed, then everything passes. The old 153 tests are untouched, because `submitted_at` is defaulted.

- [ ] **Step 6: Commit**

```bash
git add agent/broker/base.py agent/broker/alpaca.py tests/test_broker_orders.py
git commit -m "Broker: read-only recent_orders and OrderInfo.submitted_at"
```

---

### Task 3: Tools, part 1: helpers, log parsing, account, trades, notes, clock, schedule

**Files:**
- Create: `agent/discord_bot/tools.py`
- Test: `tests/test_discord_tools.py`

**Interfaces:**
- Consumes: `PaperBook(path)` with `.get(key, default)`, `.logs(limit) -> list[dict(day, kind, text)]` and `.close()`; a broker with `get_account() -> AccountInfo`, `get_positions() -> list[PositionInfo]`, `get_clock() -> MarketClock` and `recent_orders(limit) -> list[OrderInfo]`.
- Produces (used by Tasks 4 to 9):
  - `ET`, re-exported from `agent.stan.zerodte.bars`
  - `money(x: float | None) -> str`, e.g. `"$61,250.10"`, `"-$5.00"`, `"unknown"`
  - `nice_day(d: date) -> str`, e.g. `"Nov 17"`
  - `clock_time(dt: datetime) -> str`, e.g. `"9:35 AM"` (in New York time if `dt` is time-zone aware)
  - `result(facts: list[str], data: dict | None = None) -> dict`
  - `LogEntry(day: date, exit_code: int | None, lines: list[str])` (frozen dataclass); `last_log_entry(path: Path) -> LogEntry | None`
  - `ToolContext(journal_path: str = "data/stan.db", zerodte_db: str = "data/zerodte.db", data_dir: Path = Path("data"), broker_factory: Callable[[], Any] | None = None, task_reader: Callable[[], list[dict]] | None = None, now: Callable[[], datetime] = <New York now>)` with method `broker()`, created lazily once
  - Tool functions, each `(ctx, ...) -> dict`: `account(ctx)`, `recent_trades(ctx, limit=10)`, `trading_log(ctx, limit=10)`, `market_clock(ctx)`, `schedule(ctx)`
  - `recent_trades` data: `{"orders": [{"day": "YYYY-MM-DD" | None, "symbol", "side", "qty", "status", "fact"}]}`
  - `trading_log` data: `{"notes": [{"day": "YYYY-MM-DD", "kind", "text"}]}`, newest first
  - `account` data: `{"equity", "cash", "day_change", "positions": [{"symbol", "qty", "value"}], "strategy", "halted", "halt_reason"}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_discord_tools.py`:

```python
"""Stan's read-only Discord tools, against a temporary journal and a fake broker."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from agent.broker.base import AccountInfo, MarketClock, OrderInfo, PositionInfo
from agent.discord_bot import tools as T
from agent.stan.paper import PaperBook

NOW = datetime(2026, 11, 17, 10, 0, tzinfo=T.ET)


class FakeBroker:
    def __init__(self, orders=None, open_=True):
        self.orders = orders if orders is not None else []
        self.open_ = open_

    def get_account(self):
        return AccountInfo("a", "ACTIVE", True, 100412.37, 61250.10, 61250.10, None, 3, False, False, "1",
                           last_equity=100000.0)

    def get_positions(self):
        return [PositionInfo("SPY", "us_equity", 52.31, 650.0, 34711.40, 100.0),
                PositionInfo("QQQ", "us_equity", 6.62, 660.0, 4450.87, 20.0)]

    def get_clock(self):
        return MarketClock(NOW, self.open_, datetime(2026, 11, 18, 14, 30, tzinfo=timezone.utc),
                           datetime(2026, 11, 17, 21, 0, tzinfo=timezone.utc))

    def recent_orders(self, limit=20):
        return self.orders[:limit]


@pytest.fixture
def ctx(tmp_path):
    journal = tmp_path / "stan.db"
    book = PaperBook(journal)
    book.set("strategy", "medium")
    book.log("2026-11-16", "thought", "Holdings are within 5% of the plan, so no trade is needed.")
    book.log("2026-11-17", "progress", "Buy 52.31 SPY (about $34,700).")
    book.close()
    return T.ToolContext(journal_path=str(journal), zerodte_db=str(tmp_path / "zerodte.db"), data_dir=tmp_path,
                         broker_factory=FakeBroker, task_reader=lambda: [], now=lambda: NOW)


def test_formatting_helpers():
    assert T.money(61250.1) == "$61,250.10" and T.money(-5) == "-$5.00" and T.money(None) == "unknown"
    assert T.nice_day(date(2026, 11, 7)) == "Nov 7"
    assert T.clock_time(datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc)) == "9:35 AM"
    assert T.result(["a"]) == {"facts": ["a"], "data": {}}


def test_account_facts(ctx):
    r = T.account(ctx)
    facts = " ".join(r["facts"])
    assert "Equity is $100,412.37." in facts and "Cash is $61,250.10." in facts
    assert "Today the account is up 0.41%." in facts
    assert "52.31 SPY worth $34,711.40" in facts and "6.62 QQQ worth $4,450.87" in facts
    assert "the medium-risk tier" in facts and "Trading is on (not paused)." in facts and "paper money" in facts
    assert r["data"]["strategy"] == "medium" and r["data"]["positions"][0]["symbol"] == "SPY"


def test_account_shows_a_pause_with_its_reason(ctx):
    book = PaperBook(ctx.journal_path)
    book.set("halted", True)
    book.set("halt_reason", "Paused by Zy")
    book.close()
    assert "Trading is paused (Paused by Zy)." in T.account(ctx)["facts"]


def test_recent_trades_filled_and_waiting(ctx):
    t = datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc)
    ctx.broker_factory = lambda: FakeBroker(orders=[
        OrderInfo("1", "SPY", "buy", 52.31, "filled", 52.31, 663.57, t, t),
        OrderInfo("2", "QQQ", "sell", 3.0, "accepted", 0.0, None, None, t),
    ])
    r = T.recent_trades(ctx)
    assert r["facts"] == ["On Nov 17 Stan bought 52.31 SPY at $663.57 (filled).",
                          "On Nov 17 Stan placed an order to sell 3 QQQ; status: accepted."]
    assert r["data"]["orders"][0]["day"] == "2026-11-17"


def test_recent_trades_when_there_are_none(ctx):
    assert T.recent_trades(ctx)["facts"] == ["No orders yet."]


def test_trading_log_newest_first(ctx):
    r = T.trading_log(ctx)
    assert r["facts"][0] == "Nov 17: Buy 52.31 SPY (about $34,700)."
    assert r["data"]["notes"][1]["day"] == "2026-11-16"


def test_market_clock(ctx):
    assert T.market_clock(ctx)["facts"] == ["The market is open. It closes at 4:00 PM New York time."]
    ctx.broker_factory = lambda: FakeBroker(open_=False)
    assert T.market_clock(ctx)["facts"] == ["The market is closed. It next opens Nov 18 at 9:30 AM New York time."]


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_last_log_entry_finished_unfinished_missing(tmp_path):
    p = write(tmp_path / "open.log", "==== Mon 11/16/2026  9:35:00.54 ====\nold\nExit code: 1\n"
                                     "==== Tue 11/17/2026  9:35:01.00 ====\nUsing: python\nExit code: 0\nSent the open report\n")
    e = T.last_log_entry(p)
    assert e.day == date(2026, 11, 17) and e.exit_code == 0 and e.lines[0].startswith("==== Tue")
    cut = write(tmp_path / "study.log", "==== Tue 11/17/2026 18:00:00.10 ====\nstudying...\n")
    assert T.last_log_entry(cut).exit_code is None
    assert T.last_log_entry(tmp_path / "missing.log") is None
    assert T.last_log_entry(write(tmp_path / "empty.log", "")) is None


def test_schedule_reports_tasks_and_logs(ctx, tmp_path):
    write(tmp_path / "open.log", "==== Tue 11/17/2026  9:35:01.00 ====\nExit code: 0\n")
    write(tmp_path / "study.log", "==== Mon 11/16/2026 18:00:00.10 ====\nstudying...\n")
    ctx.task_reader = lambda: [
        {"name": "Stan open run", "state": "Ready", "last_run": "2026-11-17T09:35:00", "last_result": 0,
         "next_run": "2026-11-18T09:35:00"},
        {"name": "Stan nightly study", "state": "Ready", "last_run": "1999-11-30T00:00:00", "last_result": 267011,
         "next_run": "2026-11-17T18:00:00"},
    ]
    facts = T.schedule(ctx)["facts"]
    assert "Stan open run: last ran Nov 17 at 9:35 AM (OK); next run Nov 18 at 9:35 AM." in facts
    assert "Stan nightly study: has never run; next run Nov 17 at 6:00 PM." in facts
    assert "The trading run log was last written on Nov 17, exit code 0 (OK)." in facts
    assert "The study log was last written on Nov 16, but that run never finished (no exit code)." in facts
    assert "The data collector log has no entries yet." in facts


def test_schedule_with_no_tasks(ctx):
    assert "No Stan tasks are scheduled in Windows." in T.schedule(ctx)["facts"]
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `python -m pytest tests/test_discord_tools.py -q`
Expected: FAIL with `ImportError: cannot import name 'tools'`.

- [ ] **Step 3: Write `tools.py` with the helpers and these five tools**

Create `agent/discord_bot/tools.py`:

```python
"""Stan's read-only tools for the Discord chat.

Every tool returns {"facts": [...], "data": {...}}. `facts` are finished sentences with every
number already computed and formatted, because a small local model misstates numbers it has to
work out or reword. Nothing here can place, cancel or change an order, pause or resume trading,
change the tier, or start training (tests/test_discord_tools.py checks the source).
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from agent.stan.paper import PaperBook
from agent.stan.zerodte.bars import ET

STRATEGY_NAMES = {
    "selloff": "the selloff-bounce strategy",
    "low": "the low-risk tier",
    "medium": "the medium-risk tier",
    "high": "the high-risk tier",
}


# ---------------------------------------------------------------- formatting
def money(x: float | None) -> str:
    if x is None:
        return "unknown"
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def nice_day(d: date) -> str:
    return f"{d:%b} {d.day}"


def clock_time(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(ET)
    return dt.strftime("%I:%M %p").lstrip("0")


def qty(x: float) -> str:
    return f"{round(float(x), 4):g}"


def result(facts: list[str], data: dict | None = None) -> dict:
    return {"facts": list(facts), "data": data or {}}


# ---------------------------------------------------------------- job logs
HEADER = re.compile(r"^==== .*?(\d{1,2})/(\d{1,2})/(\d{4})")
EXIT = re.compile(r"^Exit code: (-?\d+)")


@dataclass(frozen=True)
class LogEntry:
    day: date
    exit_code: int | None   # None: the run started but never wrote its exit code
    lines: list[str]        # the last 10 lines of this run's block


def last_log_entry(path: Path) -> LogEntry | None:
    """The newest run in a job log written by the .bat files, or None if there is none."""
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if HEADER.match(line)]
    if not starts:
        return None
    block = lines[starts[-1]:]
    m = HEADER.match(block[0])
    codes = [int(x.group(1)) for x in map(EXIT.match, block) if x]
    return LogEntry(date(int(m.group(3)), int(m.group(1)), int(m.group(2))), codes[-1] if codes else None, block[-10:])


# ---------------------------------------------------------------- context
def default_broker():
    from agent.broker.alpaca import AlpacaBroker
    from agent.config import load_settings

    return AlpacaBroker(load_settings())


def read_windows_tasks() -> list[dict]:
    """Stan's scheduled tasks, read with PowerShell. Read only."""
    ps = ("Get-ScheduledTask -TaskName 'Stan*' | ForEach-Object { $i = Get-ScheduledTaskInfo -InputObject $_; "
          "[pscustomobject]@{name=$_.TaskName; state=[string]$_.State; "
          "last_run=$(if ($i.LastRunTime) { $i.LastRunTime.ToString('s') } else { '' }); "
          "last_result=$i.LastTaskResult; "
          "next_run=$(if ($i.NextRunTime) { $i.NextRunTime.ToString('s') } else { '' })} } | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         timeout=30).stdout.strip()
    if not out:
        return []
    rows = json.loads(out)
    return rows if isinstance(rows, list) else [rows]


@dataclass
class ToolContext:
    journal_path: str = "data/stan.db"
    zerodte_db: str = "data/zerodte.db"
    data_dir: Path = Path("data")
    broker_factory: Callable[[], Any] | None = None
    task_reader: Callable[[], list[dict]] | None = None
    now: Callable[[], datetime] = field(default=lambda: datetime.now(ET))
    _broker: Any = field(default=None, init=False, repr=False)
    _factory_used: Any = field(default=None, init=False, repr=False)

    def broker(self):
        factory = self.broker_factory or default_broker
        if self._broker is None or self._factory_used is not factory:
            self._broker, self._factory_used = factory(), factory
        return self._broker


# ---------------------------------------------------------------- tools
def account(ctx: ToolContext) -> dict:
    b = ctx.broker()
    acct, positions = b.get_account(), b.get_positions()
    book = PaperBook(ctx.journal_path)
    try:
        strategy = book.get("strategy", "selloff")
        halted, reason = bool(book.get("halted", False)), book.get("halt_reason")
    finally:
        book.close()
    day = (acct.equity / acct.last_equity - 1) if acct.last_equity else None
    held = [p for p in positions if p.qty]
    facts = [f"Equity is {money(acct.equity)}.", f"Cash is {money(acct.cash)}."]
    if day is not None:
        facts.append(f"Today the account is {'up' if day >= 0 else 'down'} {abs(day) * 100:.2f}%.")
    if held:
        facts.append("Holding " + ", ".join(f"{qty(p.qty)} {p.symbol} worth {money(p.market_value)}" for p in held) + ".")
    else:
        facts.append("No positions: the account is all cash.")
    facts.append(f"The paper account trades {STRATEGY_NAMES.get(strategy, strategy)}.")
    if halted:
        facts.append(f"Trading is paused ({reason})." if reason else "Trading is paused.")
    else:
        facts.append("Trading is on (not paused).")
    facts.append("This is paper money, not real money.")
    return result(facts, {
        "equity": acct.equity, "cash": acct.cash, "day_change": day,
        "positions": [{"symbol": p.symbol, "qty": p.qty, "value": p.market_value} for p in held],
        "strategy": strategy, "halted": halted, "halt_reason": reason,
    })


def recent_trades(ctx: ToolContext, limit: int = 10) -> dict:
    lister = getattr(ctx.broker(), "recent_orders", None)
    if lister is None:
        return result(["This broker can't list orders."])
    orders = lister(max(1, min(int(limit), 20)))
    if not orders:
        return result(["No orders yet."], {"orders": []})
    rows = []
    for o in orders:
        when = o.filled_at or o.submitted_at
        d = when.astimezone(ET).date() if when else None
        on = f"On {nice_day(d)}" if d else "On an unknown day"
        if o.status == "filled":
            verb = "bought" if o.side == "buy" else "sold"
            fact = f"{on} Stan {verb} {qty(o.filled_qty)} {o.symbol} at {money(o.filled_avg_price)} (filled)."
        else:
            fact = f"{on} Stan placed an order to {o.side} {qty(o.qty)} {o.symbol}; status: {o.status.replace('_', ' ')}."
        rows.append({"day": d.isoformat() if d else None, "symbol": o.symbol, "side": o.side, "qty": o.qty,
                     "status": o.status, "fact": fact})
    return result([r["fact"] for r in rows], {"orders": rows})


def trading_log(ctx: ToolContext, limit: int = 10) -> dict:
    book = PaperBook(ctx.journal_path)
    try:
        rows = book.logs(max(1, min(int(limit), 30)))
    finally:
        book.close()
    if not rows:
        return result(["No trading notes yet."], {"notes": []})
    notes = [{"day": r["day"], "kind": r["kind"], "text": r["text"]} for r in rows]
    return result([f"{nice_day(date.fromisoformat(n['day']))}: {n['text']}" for n in notes], {"notes": notes})


def market_clock(ctx: ToolContext) -> dict:
    c = ctx.broker().get_clock()
    if c.is_open:
        fact = f"The market is open. It closes at {clock_time(c.next_close)} New York time."
    else:
        nxt = c.next_open.astimezone(ET)
        fact = f"The market is closed. It next opens {nice_day(nxt.date())} at {clock_time(nxt)} New York time."
    return result([fact], {"is_open": c.is_open, "next_open": c.next_open.isoformat(),
                           "next_close": c.next_close.isoformat()})


JOB_LOGS = (("trading run", "open.log"), ("study", "study.log"), ("data collector", "collect.log"))
RESULT_WORDS = {0: "OK", 267009: "still running", 267011: "has not run yet"}


def _when(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(raw) if raw else None
    except ValueError:
        return None
    return None if dt is None or dt.year < 2000 else dt


def schedule(ctx: ToolContext) -> dict:
    tasks = (ctx.task_reader or read_windows_tasks)()
    facts = []
    for t in sorted(tasks, key=lambda r: r.get("name", "")):
        last, nxt = _when(t.get("last_run", "")), _when(t.get("next_run", ""))
        code = t.get("last_result")
        ran = (f"last ran {nice_day(last.date())} at {clock_time(last)} "
               f"({RESULT_WORDS.get(code, f'error code {code}')})") if last else "has never run"
        upcoming = f"next run {nice_day(nxt.date())} at {clock_time(nxt)}" if nxt else "no next run scheduled"
        facts.append(f"{t.get('name')}: {ran}; {upcoming}.")
    if not tasks:
        facts.append("No Stan tasks are scheduled in Windows.")
    for label, fname in JOB_LOGS:
        e = last_log_entry(Path(ctx.data_dir) / fname)
        if e is None:
            facts.append(f"The {label} log has no entries yet.")
        elif e.exit_code is None:
            facts.append(f"The {label} log was last written on {nice_day(e.day)}, but that run never finished (no exit code).")
        else:
            ok = "OK" if e.exit_code == 0 else "failed"
            facts.append(f"The {label} log was last written on {nice_day(e.day)}, exit code {e.exit_code} ({ok}).")
    return result(facts, {"tasks": tasks})
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python -m pytest tests/test_discord_tools.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/discord_bot/tools.py tests/test_discord_tools.py
git commit -m "Discord tools: account, trades, notes, clock, schedule"
```

---

### Task 4: Tools, part 2: school, exams, lessons, strategy

**Files:**
- Modify: `agent/discord_bot/tools.py` (add four tools at the end)
- Test: `tests/test_discord_tools.py` (append)

**Interfaces:**
- Consumes: `Journal(path)` with `.meta() -> dict`, `.exams() -> list[dict]`, `.lessons() -> list[dict]` and `.close()`. Meta keys: `run_id` (`"YYYYMMDD-..."`), `state` (`"training"` or `"finished"`), `mode` (`"real"` or `"demo"`), `summary` (`stan_growth`, `hold_growth`, `trades`, `win_rate`, `years_tested`, `years_passed`, `years_beat_hold`, `max_dd`, `hold_max_dd`, `ready_for_paper`), and `arena` (`winners: {low, medium, high} -> key`, `strategies: [{key, name, cagr, max_dd}]`). `PaperBook` meta: `strategy`, `tier_plan` (`{tier, reason, day}`).
- Produces: `study_report(ctx)`, `exams(ctx, limit=15)`, `lessons(ctx, limit=10)` (newest first), `strategy(ctx)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_discord_tools.py`:

```python
from agent.stan.journal import Journal


def seed_journal(path, state="finished", mode="real"):
    j = Journal(path)
    j.set_meta("run_id", "20261116-231341-818600")
    j.set_meta("state", state)
    j.set_meta("mode", mode)
    j.set_meta("summary", {"stan_growth": 1.12, "hold_growth": 6.29, "trades": 8818, "win_rate": 0.575,
                           "years_tested": 11, "years_passed": 9, "years_beat_hold": 2, "max_dd": -0.1784,
                           "hold_max_dd": -0.3514, "ready_for_paper": False})
    j.set_meta("arena", {"winners": {"low": "vt_s", "medium": "vt", "high": "lev"}, "strategies": [
        {"key": "vt_s", "name": "Volatility target at 73% size", "cagr": 0.0972, "max_dd": -0.1844},
        {"key": "vt", "name": "Volatility target", "cagr": 0.1248, "max_dd": -0.2459},
        {"key": "lev", "name": "3x QQQ with a 200-day trend switch", "cagr": 0.3131, "max_dd": -0.6128}]})
    j.record_exam({"year": 2016, "n_calls": 100, "accuracy": 0.52, "baseline_up": 0.53, "trades": 753,
                   "win_rate": 0.5976, "profit_factor": 1.26, "stan_return": 0.0382, "hold_return": 0.2513,
                   "passed": True, "beat_hold": False})
    j.add_lesson("mistake", "I do worse when the stock is overbought", "Across 373 trades I won 53.3%.")
    j.add_lesson("strength", "I do best after a sharp selloff", "Across 291 trades I won 69.4%.")
    j.close()


def test_study_report_facts(ctx):
    seed_journal(ctx.journal_path)
    facts = T.study_report(ctx)["facts"]
    assert facts[0] == "Last study term: Nov 16."
    assert "Verdict: not ready yet, because simply holding beat him." in facts
    assert "$1 with Stan became $1.12. $1 simply held became $6.29. Holding did better by $5.17." in facts
    assert "He won 57.5% of 8,818 practice trades." in facts
    assert "He passed 9 of 11 yearly exams and beat simply holding in 2 of them." in facts
    assert "His worst drop was -17.8%; simply holding's worst drop was -35.1%." in facts
    assert "Real money stays locked until the go-live checks pass." in facts


def test_study_report_while_in_class_and_in_demo_mode(ctx):
    seed_journal(ctx.journal_path, state="training", mode="demo")
    facts = T.study_report(ctx)["facts"]
    assert "He is in class right now, so these are the results of the last finished term." in facts
    assert "These results come from made-up practice data, not real market history." in facts


def test_study_report_before_any_term(ctx):
    assert T.study_report(ctx)["facts"] == ["Stan hasn't finished a study term yet."]


def test_exams_and_lessons(ctx):
    seed_journal(ctx.journal_path)
    assert T.exams(ctx)["facts"] == [
        "2016: passed. Stan +3.8% vs simply holding +25.1%; 753 trades, won 59.8%."]
    assert T.lessons(ctx)["facts"] == [
        "Strength: I do best after a sharp selloff. Across 291 trades I won 69.4%.",
        "Mistake: I do worse when the stock is overbought. Across 373 trades I won 53.3%."]
    assert len(T.lessons(ctx, limit=1)["facts"]) == 1


def test_strategy_facts(ctx):
    seed_journal(ctx.journal_path)
    book = PaperBook(ctx.journal_path)
    book.set("tier_plan", {"tier": "medium", "reason": "hold 100% SPY", "day": "2026-11-13"})
    book.close()
    facts = T.strategy(ctx)["facts"]
    assert facts[0] == "The paper account trades the medium-risk tier."
    assert "Best medium-risk strategy: Volatility target, 12.5% a year, worst drop -24.6%." in facts
    assert "Best high-risk strategy: 3x QQQ with a 200-day trend switch, 31.3% a year, worst drop -61.3%." in facts
    assert "Latest plan (Nov 13): hold 100% SPY." in facts
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `python -m pytest tests/test_discord_tools.py -q`
Expected: the 5 new tests FAIL with `AttributeError: module 'agent.discord_bot.tools' has no attribute 'study_report'`.

- [ ] **Step 3: Add the four tools**

Add `from agent.stan.journal import Journal` to the imports in `agent/discord_bot/tools.py`, and append:

```python
def _journal_meta(ctx: ToolContext) -> dict:
    j = Journal(ctx.journal_path)
    try:
        return j.meta()
    finally:
        j.close()


def study_report(ctx: ToolContext) -> dict:
    meta = _journal_meta(ctx)
    s = meta.get("summary")
    if not s:
        return result(["Stan hasn't finished a study term yet."])
    facts = []
    run = str(meta.get("run_id", ""))
    if re.match(r"\d{8}", run):
        facts.append(f"Last study term: {nice_day(datetime.strptime(run[:8], '%Y%m%d').date())}.")
    if meta.get("state") == "training":
        facts.append("He is in class right now, so these are the results of the last finished term.")
    if meta.get("mode") == "demo":
        facts.append("These results come from made-up practice data, not real market history.")
    stan, hold = s.get("stan_growth"), s.get("hold_growth")
    if s.get("ready_for_paper"):
        facts.append("Verdict: he passed school.")
    elif stan is not None and hold is not None and hold > stan:
        facts.append("Verdict: not ready yet, because simply holding beat him.")
    else:
        facts.append("Verdict: not ready yet.")
    if stan is not None and hold is not None:
        better = "Holding" if hold > stan else "Stan"
        facts.append(f"$1 with Stan became {money(stan)}. $1 simply held became {money(hold)}. "
                     f"{better} did better by {money(abs(hold - stan))}.")
    if s.get("trades") is not None and s.get("win_rate") is not None:
        facts.append(f"He won {s['win_rate'] * 100:.1f}% of {s['trades']:,} practice trades.")
    if s.get("years_tested"):
        facts.append(f"He passed {s.get('years_passed', 0)} of {s['years_tested']} yearly exams and beat simply "
                     f"holding in {s.get('years_beat_hold', 0)} of them.")
    if s.get("max_dd") is not None and s.get("hold_max_dd") is not None:
        facts.append(f"His worst drop was {s['max_dd'] * 100:.1f}%; simply holding's worst drop was "
                     f"{s['hold_max_dd'] * 100:.1f}%.")
    facts.append("Real money stays locked until the go-live checks pass.")
    return result(facts, {"summary": s, "state": meta.get("state"), "mode": meta.get("mode")})


def exams(ctx: ToolContext, limit: int = 15) -> dict:
    j = Journal(ctx.journal_path)
    try:
        rows = j.exams()[-max(1, min(int(limit), 20)):]
    finally:
        j.close()
    if not rows:
        return result(["Stan hasn't sat any yearly exams yet."], {"exams": []})
    facts = [f"{e['year']}: {'passed' if e['passed'] else 'failed'}. Stan {e['stan_return'] * 100:+.1f}% vs simply "
             f"holding {e['hold_return'] * 100:+.1f}%; {e['trades']:,} trades, won {e['win_rate'] * 100:.1f}%."
             for e in rows]
    return result(facts, {"exams": rows})


def lessons(ctx: ToolContext, limit: int = 10) -> dict:
    j = Journal(ctx.journal_path)
    try:
        rows = list(reversed(j.lessons()))[:max(1, min(int(limit), 20))]
    finally:
        j.close()
    if not rows:
        return result(["Stan hasn't written any lessons yet."], {"lessons": []})
    facts = [f"{'Mistake' if r['kind'] == 'mistake' else 'Strength'}: {r['title']}. {r['text']}" for r in rows]
    return result(facts, {"lessons": [{"kind": r["kind"], "title": r["title"]} for r in rows]})


def strategy(ctx: ToolContext) -> dict:
    book = PaperBook(ctx.journal_path)
    try:
        name, plan = book.get("strategy", "selloff"), book.get("tier_plan")
    finally:
        book.close()
    facts = [f"The paper account trades {STRATEGY_NAMES.get(name, name)}."]
    arena = _journal_meta(ctx).get("arena") or {}
    by_key = {x.get("key"): x for x in arena.get("strategies", [])}
    for tier in ("low", "medium", "high"):
        w = by_key.get((arena.get("winners") or {}).get(tier))
        if w and w.get("cagr") is not None and w.get("max_dd") is not None:
            facts.append(f"Best {tier}-risk strategy: {w['name']}, {w['cagr'] * 100:.1f}% a year, "
                         f"worst drop {w['max_dd'] * 100:.1f}%.")
    if plan and plan.get("reason"):
        day = f" ({nice_day(date.fromisoformat(plan['day']))})" if plan.get("day") else ""
        facts.append(f"Latest plan{day}: {plan['reason']}.")
    return result(facts, {"strategy": name, "tier_plan": plan})
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python -m pytest tests/test_discord_tools.py -q`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/discord_bot/tools.py tests/test_discord_tools.py
git commit -m "Discord tools: study report, exams, lessons, strategy"
```

---

### Task 5: Tools, part 3: levels, the registry, and the read-only check; then the brain

This task finishes `tools.py` with the registry that the brain needs, then builds `brain.py`.

**Files:**
- Modify: `agent/discord_bot/tools.py` (append `levels`, `Tool`, `TOOLS`, `tool_specs`, `run_tool`)
- Create: `agent/discord_bot/brain.py`
- Test: `tests/test_discord_tools.py` (append), `tests/test_discord_brain.py` (create)

**Interfaces:**
- Consumes: `agent.stan.zerodte.collect.today_view(store, cfg, day) -> dict | None`, `collect.report(view, day) -> str`, `ZeroStore(path)` with `.snapshot_days(symbol) -> dict[str, int]` and `.close()`, and `ZeroDteConfig().symbol`. `agent.stan.persona.STAN` (keys `name`, `title`, `goal`, `bio`, `principles`).
- Produces:
  - `levels(ctx, day: str | None = None) -> dict`
  - `tool_specs() -> list[dict]` in Ollama's format: `{"type": "function", "function": {"name", "description", "parameters"}}`
  - `run_tool(ctx, name: str, args: dict | None) -> dict`. It never raises: an unknown tool or an exception becomes `{"error": "..."}`, and unknown argument names are dropped.
  - `brain.system_prompt() -> str`
  - `brain.Brain(ctx, model="qwen3:14b", url="http://127.0.0.1:11434", thinking=False, post=None, timeout=300.0)` with `.reply(history: list[dict], message: str) -> str`
  - `brain.NOT_RUNNING`, `brain.EMPTY` (string constants); `brain.MAX_TOOL_CALLS = 8`; `brain.NUM_CTX = 8192`

- [ ] **Step 1: Write the failing tool tests**

Append to `tests/test_discord_tools.py`:

```python
import ast

from agent.discord_bot import brain as Bmod


def test_levels_without_any_snapshot(ctx):
    assert T.levels(ctx)["facts"] == ["No QQQ options snapshot is saved for Nov 17."]


def test_levels_uses_the_latest_saved_day(ctx, monkeypatch):
    from agent.stan.zerodte import collect

    seen = []

    def fake_view(store, cfg, day):
        seen.append(day)
        return {"day": day} if day == date(2026, 11, 16) else None

    monkeypatch.setattr(collect, "today_view", fake_view)
    monkeypatch.setattr(collect, "report", lambda view, day: "QQQ 661.63 at 2026-11-16 23:01 ET\n\nLevels:\n  660.00  below")
    monkeypatch.setattr("agent.stan.zerodte.store.ZeroStore.snapshot_days", lambda self, sym: {"2026-11-16": 1})
    r = T.levels(ctx)
    assert seen == [date(2026, 11, 17), date(2026, 11, 16)]
    assert r["facts"] == ["QQQ 661.63 at 2026-11-16 23:01 ET", "Levels:", "660.00  below"]


def test_registry_specs_and_errors(ctx):
    names = [s["function"]["name"] for s in T.tool_specs()]
    assert names == ["account", "recent_trades", "trading_log", "study_report", "exams", "lessons", "strategy",
                     "levels", "schedule", "market_clock"]
    assert "school" in next(s for s in T.tool_specs() if s["function"]["name"] == "study_report")["function"]["description"]
    assert T.run_tool(ctx, "place_order", {}) == {"error": "There is no tool called place_order."}
    assert T.run_tool(ctx, "trading_log", {"limit": 1, "symbol": "SPY"})["facts"] == ["Nov 17: Buy 52.31 SPY (about $34,700)."]
    ctx.broker_factory = lambda: (_ for _ in ()).throw(ConnectionError("Alpaca down"))
    assert T.run_tool(ctx, "account", {}) == {"error": "ConnectionError: Alpaca down"}


FORBIDDEN = {"submit_open_order", "submit_fractional_order", "cancel_open_orders", "safe_submit", "run_evening",
             "run_tier", "sell_all", "set_strategy", "run_training", "record_equity", "add_trade", "set_meta",
             "think", "set", "Popen", "train"}


def test_discord_package_is_read_only():
    pkg = Path(T.__file__).parent
    for src in pkg.glob("*.py"):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN, f"{src.name} uses .{node.attr}"
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("agent.stan.train", "agent.stan.tiers"), f"{src.name} imports {node.module}"
                assert not {a.name for a in node.names} & FORBIDDEN, f"{src.name} imports a write function"
```

- [ ] **Step 2: Write the failing brain tests**

Create `tests/test_discord_brain.py`:

```python
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
    assert json.loads(tool_msg["content"]) == {"facts": ["account fact"], "data": {}}


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
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `python -m pytest tests/test_discord_tools.py tests/test_discord_brain.py -q`
Expected: FAIL with `ImportError: cannot import name 'brain'`. After `brain.py` exists, the levels and registry tests fail on missing `levels`, `tool_specs` and `run_tool`.

- [ ] **Step 4: Finish `tools.py`**

Append to `agent/discord_bot/tools.py`:

```python
def levels(ctx: ToolContext, day: str | None = None) -> dict:
    from agent.stan.zerodte import collect
    from agent.stan.zerodte.config import ZeroDteConfig
    from agent.stan.zerodte.store import ZeroStore

    cfg = ZeroDteConfig()
    store = ZeroStore(ctx.zerodte_db)
    try:
        target = date.fromisoformat(day) if day else ctx.now().date()
        view = collect.today_view(store, cfg, target)
        if view is None and not day:
            saved = sorted(store.snapshot_days(cfg.symbol))
            if saved and date.fromisoformat(saved[-1]) != target:
                target = date.fromisoformat(saved[-1])
                view = collect.today_view(store, cfg, target)
        if view is None:
            return result([f"No QQQ options snapshot is saved for {nice_day(target)}."])
        lines = [line.strip() for line in collect.report(view, target).splitlines() if line.strip()]
        return result(lines[:25], {"day": target.isoformat()})
    finally:
        store.close()


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict
    func: Callable[..., dict]


def _params(**props) -> dict:
    return {"type": "object", "properties": props}


LIMIT = {"type": "integer", "description": "How many to show (optional)."}

TOOLS = [
    Tool("account", "Right now in the paper account: equity, cash, today's change, what Stan holds (positions), "
         "which risk tier it trades, and whether trading is paused. Use for money, balance, cash, holdings, "
         "'how much do I have', 'how's the account', paused questions.", _params(), account),
    Tool("recent_trades", "What Stan bought or sold recently: his latest orders with day, symbol, side, quantity, "
         "price and status. Use for 'what did you buy', 'what did you sell', 'any trades today'.",
         _params(limit=LIMIT), recent_trades),
    Tool("trading_log", "Stan's own notes from his recent trading runs, including why he did or didn't trade. "
         "Use for 'why didn't you trade', 'what happened this morning', 'what's the plan today'.",
         _params(limit=LIMIT), trading_log),
    Tool("study_report", "Stan's school: his last study term (he studies market history every evening). Verdict, "
         "report card, practice-trade results and whether his strategy beats simply holding. Use for school, "
         "studying, class, learning, report card, readiness, and real money questions.", _params(), study_report),
    Tool("exams", "Stan's yearly exams from school: for each year, passed or failed, his return vs simply holding. "
         "Use for 'how did you do in 2022', 'which years did you fail'.", _params(limit=LIMIT), exams),
    Tool("lessons", "The lessons Stan wrote from his wins and losses (mistakes and strengths). Use for "
         "'what have you learned', 'what are your weaknesses', 'what are you good at'.",
         _params(limit=LIMIT), lessons),
    Tool("strategy", "Which strategy the paper account trades, the best strategy in each risk tier (low, medium, "
         "high), and the latest plan. Use for 'what are you trading', 'why SPY', 'which tier'.", _params(), strategy),
    Tool("levels", "The QQQ 0DTE options levels for a day: price levels, GEX (gamma exposure), call and put walls, "
         "and the put/call table. Use for levels, GEX, walls, put/call, QQQ options questions.",
         _params(day={"type": "string", "description": "A date like 2026-11-17 (optional, default today)."}), levels),
    Tool("schedule", "Stan's automatic jobs: when each scheduled task last ran and runs next, and whether each "
         "job's last run finished OK. Use for 'did everything run', 'did you trade this morning', 'is school "
         "tonight', 'is something broken'.", _params(), schedule),
    Tool("market_clock", "Whether the stock market is open now, and when it next opens or closes. Use for "
         "'is the market open', 'when does it open'.", _params(), market_clock),
]
BY_NAME = {t.name: t for t in TOOLS}


def tool_specs() -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.params}}
            for t in TOOLS]


def run_tool(ctx: ToolContext, name: str, args: dict | None) -> dict:
    tool = BY_NAME.get(name)
    if tool is None:
        return {"error": f"There is no tool called {name}."}
    allowed = set(tool.params.get("properties", {}))
    kwargs = {k: v for k, v in (args or {}).items() if k in allowed and v not in (None, "")}
    try:
        return tool.func(ctx, **kwargs)
    except Exception as err:  # one bad data source must not break the whole answer
        return {"error": f"{type(err).__name__}: {err}"}
```

- [ ] **Step 5: Write `brain.py`**

Create `agent/discord_bot/brain.py`:

```python
"""Stan's brain for Discord: a free local model in Ollama that answers from read-only tools.

Run one question from a terminal (needs Ollama, not Discord):
    python -m agent.discord_bot.brain "how did today go?"
"""
from __future__ import annotations

import json
import re
import sys

import httpx

from agent.discord_bot.tools import ToolContext, run_tool, tool_specs
from agent.stan.persona import STAN

MAX_TOOL_CALLS = 8
NUM_CTX = 8192
NOT_RUNNING = "My brain isn't running. Start Ollama on the PC and ask again."
EMPTY = "I couldn't finish that answer. Try asking another way."

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
        payload = {"model": self.model, "messages": messages, "stream": False, "think": self.thinking,
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
                    messages.append({"role": "tool", "tool_name": name, "content": json.dumps(out)})
            return EMPTY
        except ModelMissing:
            return f"My brain model isn't downloaded yet. On the PC, run: ollama pull {self.model}"
        except httpx.TimeoutException:
            return "My brain took too long to answer. Try again in a minute."
        except httpx.TransportError:
            return NOT_RUNNING
        except httpx.HTTPStatusError as err:
            return f"My brain had a problem (HTTP {err.response.status_code}). Try again in a minute."


def main(argv: list[str] | None = None) -> int:
    from agent.discord_bot.config import load_chat_settings

    question = " ".join(argv if argv is not None else sys.argv[1:]).strip()
    if not question:
        print('Usage: python -m agent.discord_bot.brain "your question"')
        return 2
    s = load_chat_settings(require_discord=False)
    print(Brain(ToolContext(journal_path=s.journal_path), s.model, s.ollama_url, s.thinking).reply([], question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to make sure they pass**

Run: `python -m pytest tests/test_discord_tools.py tests/test_discord_brain.py -q`
Expected: 19 + 8 = 27 passed.

If `test_discord_package_is_read_only` fails on `.set` or `.think` inside the package, the fix is to not call those methods. These tests never need them.

- [ ] **Step 7: Commit**

```bash
git add agent/discord_bot/tools.py agent/discord_bot/brain.py tests/test_discord_tools.py tests/test_discord_brain.py
git commit -m "Discord chat: levels tool, tool registry, read-only check, and Stan's Ollama brain"
```

---

### Task 6: Reports and the missed-run check

**Files:**
- Create: `agent/discord_bot/reports.py`
- Test: `tests/test_discord_reports.py`

**Interfaces:**
- Consumes: `tools.run_tool`, `tools.last_log_entry`, `tools.nice_day`, `tools.money`, `tools.ToolContext` (`now`, `data_dir`)
- Produces:
  - `split_message(text: str, limit: int = 2000) -> list[str]`
  - `JOB_NAMES: dict[str, str]`, `LOG_FILES: dict[str, str]` (keys `"open"`, `"study"`, `"collect"`)
  - `job_report(ctx, job: str, code: int) -> str | None` (`None` means nothing to send)
  - `missed_run_message(job: str, data_dir: Path, today: date) -> str | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_discord_reports.py`:

```python
"""Fixed-wording reports, message splitting and the missed-run check."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from agent.discord_bot import reports as R
from agent.discord_bot import tools as T

NOW = datetime(2026, 11, 17, 9, 40, tzinfo=T.ET)


def fake_tools(monkeypatch, answers):
    monkeypatch.setattr(R, "run_tool", lambda ctx, name, args: answers[name])


@pytest.fixture
def ctx(tmp_path):
    return T.ToolContext(data_dir=tmp_path, now=lambda: NOW)


def test_split_message_on_lines_and_hard_cuts():
    assert R.split_message("") == []
    assert R.split_message("short") == ["short"]
    parts = R.split_message("a" * 1500 + "\n" + "b" * 1500)
    assert parts == ["a" * 1500, "b" * 1500]
    long = R.split_message("x" * 4500)
    assert [len(p) for p in long] == [2000, 2000, 500]
    assert all(len(p) <= 2000 for p in R.split_message(("word " * 900).strip()))


ACCOUNT = {"facts": [], "data": {"equity": 100412.37, "day_change": 0.0041, "halted": False}}


def test_open_report_with_orders_today(ctx, monkeypatch):
    fake_tools(monkeypatch, {
        "recent_trades": {"facts": [], "data": {"orders": [
            {"day": "2026-11-17", "fact": "On Nov 17 Stan bought 52.31 SPY at $663.57 (filled)."},
            {"day": "2026-11-13", "fact": "On Nov 13 Stan bought 1 QQQ at $600.00 (filled)."}]}},
        "trading_log": {"facts": [], "data": {"notes": []}},
        "account": ACCOUNT})
    assert R.job_report(ctx, "open", 0) == (
        "Trading run for Nov 17:\n"
        "On Nov 17 Stan bought 52.31 SPY at $663.57 (filled).\n"
        "Equity is $100,412.37, up 0.41% today. Paper money only.")


def test_open_report_without_orders_uses_todays_notes(ctx, monkeypatch):
    fake_tools(monkeypatch, {
        "recent_trades": {"facts": ["No orders yet."], "data": {"orders": []}},
        "trading_log": {"facts": [], "data": {"notes": [
            {"day": "2026-11-17", "text": "Holdings are within 5% of the plan, so no trade is needed."},
            {"day": "2026-11-16", "text": "old note"}]}},
        "account": {"facts": [], "data": {"equity": 99000.0, "day_change": -0.01, "halted": True}}})
    assert R.job_report(ctx, "open", 0) == (
        "Trading run for Nov 17:\n"
        "No orders today. Holdings are within 5% of the plan, so no trade is needed.\n"
        "Trading is paused, so no orders go out until it's switched back on in the web app.\n"
        "Equity is $99,000.00, down 1.00% today. Paper money only.")


def test_open_report_when_alpaca_is_down(ctx, monkeypatch):
    fake_tools(monkeypatch, {"recent_trades": {"error": "ConnectionError: down"},
                             "trading_log": {"facts": [], "data": {"notes": []}},
                             "account": {"error": "ConnectionError: down"}})
    text = R.job_report(ctx, "open", 0)
    assert "I couldn't read today's orders (ConnectionError: down)." in text
    assert "I couldn't read the account (ConnectionError: down)." in text


def test_study_report(ctx, monkeypatch):
    fake_tools(monkeypatch, {
        "study_report": {"facts": ["Last study term: Nov 17.", "Verdict: not ready yet."], "data": {}},
        "lessons": {"facts": ["Strength: I do best after a sharp selloff."], "data": {}}})
    assert R.job_report(ctx, "study", 0) == (
        "School's out for today.\nLast study term: Nov 17.\nVerdict: not ready yet.\n"
        "Newest lesson: Strength: I do best after a sharp selloff.")


def test_failure_report_includes_the_log_tail(ctx):
    (ctx.data_dir / "collect.log").write_text("==== Tue 11/17/2026  9:25:00.00 ====\nTraceback: boom\nExit code: 1\n")
    text = R.job_report(ctx, "collect", 1)
    assert text.startswith("Problem: the 0DTE data collector failed with exit code 1.")
    assert "Traceback: boom" in text and "data/collect.log" in text


def test_collect_success_sends_nothing(ctx):
    assert R.job_report(ctx, "collect", 0) is None


def test_unknown_job(ctx):
    with pytest.raises(ValueError):
        R.job_report(ctx, "dance", 0)


def test_missed_run_message(tmp_path):
    today = date(2026, 11, 17)
    assert "hasn't run today" in R.missed_run_message("open", tmp_path, today)
    (tmp_path / "open.log").write_text("==== Mon 11/16/2026  9:35:00.00 ====\nExit code: 0\n")
    assert "the 9:35 AM trading run hasn't run today" in R.missed_run_message("open", tmp_path, today)
    (tmp_path / "open.log").write_text("==== Tue 11/17/2026  9:35:00.00 ====\n")
    assert R.missed_run_message("open", tmp_path, today) is None
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `python -m pytest tests/test_discord_reports.py -q`
Expected: FAIL with `ImportError: cannot import name 'reports'`.

- [ ] **Step 3: Write `reports.py`**

Create `agent/discord_bot/reports.py`:

```python
"""Stan's Discord reports. Fixed wording built from the numbers, never from the AI model, so a
report can't misstate a number and still arrives when Ollama is off."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from agent.discord_bot.tools import ToolContext, last_log_entry, money, nice_day, run_tool

JOB_NAMES = {"open": "the 9:35 AM trading run", "study": "the 6 PM study term", "collect": "the 0DTE data collector"}
LOG_FILES = {"open": "open.log", "study": "study.log", "collect": "collect.log"}


def split_message(text: str, limit: int = 2000) -> list[str]:
    """Cut text into Discord-sized messages, on line breaks, then spaces, then anywhere."""
    text = (text or "").strip()
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def _failure(job: str, code: int, data_dir: Path) -> str:
    entry = last_log_entry(Path(data_dir) / LOG_FILES[job])
    tail = "\n".join(entry.lines) if entry else "(no log found)"
    return (f"Problem: {JOB_NAMES[job]} failed with exit code {code}.\n"
            f"Last lines of data/{LOG_FILES[job]}:\n```\n{tail[-1500:]}\n```")


def _open(ctx: ToolContext) -> str:
    today = ctx.now().date()
    lines = [f"Trading run for {nice_day(today)}:"]
    orders = run_tool(ctx, "recent_trades", {"limit": 20})
    if "error" in orders:
        lines.append(f"I couldn't read today's orders ({orders['error']}).")
    else:
        todays = [o["fact"] for o in orders["data"].get("orders", []) if o.get("day") == today.isoformat()]
        if todays:
            lines += todays
        else:
            notes = run_tool(ctx, "trading_log", {"limit": 10})
            texts = [n["text"] for n in notes.get("data", {}).get("notes", []) if n.get("day") == today.isoformat()]
            lines.append(" ".join(["No orders today."] + texts[:3]))
    acct = run_tool(ctx, "account", {})
    if "error" in acct:
        lines.append(f"I couldn't read the account ({acct['error']}).")
    else:
        d = acct["data"]
        if d.get("halted"):
            lines.append("Trading is paused, so no orders go out until it's switched back on in the web app.")
        change = d.get("day_change")
        moved = "" if change is None else f", {'up' if change >= 0 else 'down'} {abs(change) * 100:.2f}% today"
        lines.append(f"Equity is {money(d.get('equity'))}{moved}. Paper money only.")
    return "\n".join(lines)


def _study(ctx: ToolContext) -> str:
    r = run_tool(ctx, "study_report", {})
    if "error" in r:
        return f"School's out for today, but I couldn't read my report card ({r['error']})."
    lines = ["School's out for today."] + r["facts"]
    newest = run_tool(ctx, "lessons", {"limit": 1})
    if newest.get("facts") and "error" not in newest:
        lines.append(f"Newest lesson: {newest['facts'][0]}")
    return "\n".join(lines)


def job_report(ctx: ToolContext, job: str, code: int) -> str | None:
    if job not in JOB_NAMES:
        raise ValueError(f"unknown job {job}")
    if code != 0:
        return _failure(job, code, ctx.data_dir)
    if job == "open":
        return _open(ctx)
    if job == "study":
        return _study(ctx)
    return None


def missed_run_message(job: str, data_dir: Path, today: date) -> str | None:
    entry = last_log_entry(Path(data_dir) / LOG_FILES[job])
    if entry is not None and entry.day == today:
        return None
    return (f"Heads up: {JOB_NAMES[job]} hasn't run today (nothing in data/{LOG_FILES[job]} for "
            f"{nice_day(today)}). The PC was probably off or asleep at that time.")
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python -m pytest tests/test_discord_reports.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/discord_bot/reports.py tests/test_discord_reports.py
git commit -m "Discord chat: fixed-wording reports, message splitting, missed-run check"
```

---

### Task 7: `notify` and the `.bat` hooks

**Files:**
- Create: `agent/discord_bot/notify.py`
- Modify: `run_open.bat`, `run_study.bat`, `run_collect.bat` (the two lines after each Python run)
- Test: `tests/test_discord_reports.py` (append)

**Interfaces:**
- Consumes: `reports.job_report`, `reports.split_message`, `config.load_chat_settings`, `tools.ToolContext`
- Produces: `notify.send(token: str, channel_id: int, text: str, post=httpx.post) -> int` (the number of messages sent; raises on an HTTP error); `notify.main(argv: list[str] | None = None, ctx: ToolContext | None = None, post=None) -> int` (always 0)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_discord_reports.py`:

```python
from pathlib import Path

import httpx

from agent.discord_bot import notify as N

ROOT = Path(__file__).resolve().parents[1]


class FakePost:
    def __init__(self, status=200):
        self.calls, self.status = [], status

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        return httpx.Response(self.status, json={}, request=httpx.Request("POST", url))


def test_send_posts_each_chunk_without_pinging_anyone():
    post = FakePost()
    assert N.send("tok", 222, "a" * 2500, post=post) == 2
    url, headers, body = post.calls[0]
    assert url == "https://discord.com/api/v10/channels/222/messages"
    assert headers["Authorization"] == "Bot tok" and headers["User-Agent"].startswith("DiscordBot (")
    assert body["allowed_mentions"] == {"parse": []} and len(body["content"]) == 2000


def discord_env(monkeypatch, ready=True):
    for k in ("DISCORD_BOT_TOKEN", "DISCORD_OWNER_ID", "DISCORD_CHANNEL_ID"):
        monkeypatch.delenv(k, raising=False)
    if ready:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setenv("DISCORD_OWNER_ID", "1")
        monkeypatch.setenv("DISCORD_CHANNEL_ID", "222")
    monkeypatch.setattr("agent.discord_bot.config.load_dotenv", None)


def test_notify_skips_without_settings(monkeypatch, capsys, ctx):
    discord_env(monkeypatch, ready=False)
    assert N.main(["open", "0"], ctx=ctx, post=FakePost()) == 0
    assert "Discord not set up, skipping report." in capsys.readouterr().out


def test_notify_sends_a_failure_and_survives_a_discord_error(monkeypatch, capsys, ctx):
    discord_env(monkeypatch)
    ok = FakePost()
    assert N.main(["collect", "1"], ctx=ctx, post=ok) == 0
    assert ok.calls and "failed with exit code 1" in ok.calls[0][2]["content"]
    assert N.main(["collect", "1"], ctx=ctx, post=FakePost(status=500)) == 0
    assert "Couldn't send the Discord report" in capsys.readouterr().out


def test_notify_collect_success_sends_nothing(monkeypatch, ctx):
    discord_env(monkeypatch)
    post = FakePost()
    assert N.main(["collect", "0"], ctx=ctx, post=post) == 0 and post.calls == []


@pytest.mark.parametrize("bat,job", [("run_open.bat", "open"), ("run_study.bat", "study"),
                                     ("run_collect.bat", "collect")])
def test_bat_files_save_the_code_then_notify(bat, job):
    text = (ROOT / bat).read_text()
    assert 'set "CODE=%errorlevel%"' in text
    assert f"echo Exit code: %CODE% >> data\\{R.LOG_FILES[job]}" in text
    assert f"%PY% -m agent.discord_bot.notify {job} %CODE% >> data\\{R.LOG_FILES[job]} 2>&1" in text
    assert text.index('set "CODE=%errorlevel%"') < text.index("agent.discord_bot.notify")
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `python -m pytest tests/test_discord_reports.py -q`
Expected: FAIL with `ImportError: cannot import name 'notify'`.

- [ ] **Step 3: Write `notify.py`**

Create `agent/discord_bot/notify.py`:

```python
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
    args = parser.parse_args(argv)
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
```

- [ ] **Step 4: Hook up the three `.bat` jobs**

In `run_open.bat`, replace:

```bat
%PY% -m agent.stan.paper >> data\open.log 2>&1
echo Exit code: %errorlevel% >> data\open.log
```

with:

```bat
%PY% -m agent.stan.paper >> data\open.log 2>&1
set "CODE=%errorlevel%"
echo Exit code: %CODE% >> data\open.log
%PY% -m agent.discord_bot.notify open %CODE% >> data\open.log 2>&1
```

In `run_study.bat`, replace:

```bat
%PY% -m agent.stan.train --pace 0 >> data\study.log 2>&1
echo Exit code: %errorlevel% >> data\study.log
```

with:

```bat
%PY% -m agent.stan.train --pace 0 >> data\study.log 2>&1
set "CODE=%errorlevel%"
echo Exit code: %CODE% >> data\study.log
%PY% -m agent.discord_bot.notify study %CODE% >> data\study.log 2>&1
```

In `run_collect.bat`, replace:

```bat
%PY% -m agent.stan.zerodte.collect watch >> data\collect.log 2>&1
echo Exit code: %errorlevel% >> data\collect.log
```

with:

```bat
%PY% -m agent.stan.zerodte.collect watch >> data\collect.log 2>&1
set "CODE=%errorlevel%"
echo Exit code: %CODE% >> data\collect.log
%PY% -m agent.discord_bot.notify collect %CODE% >> data\collect.log 2>&1
```

Keep the files' CRLF line endings. After editing, check with: `file run_open.bat run_study.bat run_collect.bat` (expected: "with CRLF line terminators" for each).

- [ ] **Step 5: Run the tests to make sure they pass**

Run: `python -m pytest tests/test_discord_reports.py -q`
Expected: 16 passed.

- [ ] **Step 6: Commit**

```bash
git add agent/discord_bot/notify.py run_open.bat run_study.bat run_collect.bat tests/test_discord_reports.py
git commit -m "Discord chat: notify command and report hooks in the scheduled jobs"
```

---

### Task 8: The bot

**Files:**
- Create: `agent/discord_bot/bot.py`
- Test: `tests/test_discord_bot.py` (append)

**Interfaces:**
- Consumes: `ChatSettings`, `load_chat_settings`, `Brain.reply`, `brain.EMPTY`, `reports.split_message`, `reports.missed_run_message`, `tools.ToolContext`, `tools.ET`
- Produces: `should_answer(author_id: int, author_is_bot: bool, channel_id: int, settings: ChatSettings) -> bool`; `due_checks(now: datetime, done: set[tuple[str, date]]) -> list[str]`; `async handle_message(content: str, channel, brain, history: deque) -> None`; `StanBot(settings, brain, data_dir=Path("data"))`; `main() -> int` (0 = clean exit, 1 = crashed, so restart; 2 = settings, token or intent problem, so don't restart)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_discord_bot.py`:

```python
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
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `python -m pytest tests/test_discord_bot.py -q`
Expected: the new tests FAIL with `ImportError: cannot import name 'bot'`.

- [ ] **Step 3: Write `bot.py`**

Create `agent/discord_bot/bot.py`:

```python
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
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python -m pytest tests/test_discord_bot.py -q`
Expected: 5 + 5 = 10 passed.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: everything passes. That's the 153 existing tests, plus 2 broker tests, 19 tool tests, 8 brain tests, 16 report tests and 10 bot tests: 208 in total.

- [ ] **Step 6: Commit**

```bash
git add agent/discord_bot/bot.py tests/test_discord_bot.py
git commit -m "Discord chat: Stan's bot with owner check and missed-run alerts"
```

---

### Task 9: Keep the bot running, setup docs, and live checks

**Files:**
- Create: `run_discord.bat`
- Create: `schedule_discord.ps1`
- Modify: `schedule_trading.bat` (header comment, removal commands, one new call before the final `pause`)
- Modify: `README.md` (new section after "Automatic trading, Monday to Friday")
- Test: manual (scripts and a live chat)

**Interfaces:**
- Consumes: `python -m agent.discord_bot.bot` exit codes (0, 1, 2); `python -m agent.discord_bot.brain "<question>"`

- [ ] **Step 1: Write `run_discord.bat` (CRLF line endings)**

```bat
@echo off
REM Keeps Stan's Discord bot running. It restarts 30 seconds after a crash, and stops for good
REM if a setting, the bot token or Discord's Message Content Intent needs fixing (see data\discord.log).
cd /d "%~dp0"
if not exist data mkdir data
set "PY="
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
if not defined PY if exist data\python_path.txt set /p PY=<data\python_path.txt
if not defined PY (where py >nul 2>&1 && set "PY=py -3")
if not defined PY set "PY=python"
:loop
echo ==== %date% %time% starting Stan's Discord bot ==== >> data\discord.log
%PY% -m agent.discord_bot.bot >> data\discord.log 2>&1
set "CODE=%errorlevel%"
echo Bot exited with code %CODE% >> data\discord.log
if "%CODE%"=="2" (
  echo Not restarting: fix the problem above, then run this file again. >> data\discord.log
  exit /b 2
)
ping -n 31 127.0.0.1 >nul
goto loop
```

After writing it, run `sed -i 's/$/\r/' run_discord.bat` (if the file was written with LF), then check: `file run_discord.bat` (expected: "with CRLF line terminators").

- [ ] **Step 2: Write `schedule_discord.ps1`**

```powershell
# Registers the "Stan Discord" task: starts run_discord.bat, hidden, whenever Zy logs in.
# No time limit (Windows otherwise stops tasks after 72 hours), keeps running on battery, one copy at a time.
$ErrorActionPreference = 'Stop'
$bat = Join-Path $PSScriptRoot 'run_discord.bat'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -WindowStyle Hidden -Command `"& '$bat'`"" -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -StartWhenAvailable
Register-ScheduledTask -TaskName 'Stan Discord' -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'Stan Discord'
Write-Output 'Done. Stan Discord starts when you log in and is starting now. Log: data\discord.log'
```

- [ ] **Step 3: Call it from `schedule_trading.bat`**

In `schedule_trading.bat`, change the second header line from:

```bat
REM runs from 9:25 AM to the close, and Stan studies a new school term at 6 PM, until you remove the tasks.
```

to:

```bat
REM runs from 9:25 AM to the close, Stan studies a new school term at 6 PM, and Stan's Discord bot
REM runs whenever you're logged in, until you remove the tasks.
```

After the line `REM   schtasks /Delete /TN "Stan nightly study" /F`, add:

```bat
REM   schtasks /Delete /TN "Stan Discord" /F
```

Immediately before the final `pause`, add:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0schedule_discord.ps1"
if errorlevel 1 (echo Could not create the Discord task.)
```

- [ ] **Step 4: Add the README section**

Insert after the "Automatic trading, Monday to Friday" section's last bullet:

```markdown
## Talk to Stan on Discord

Write to Stan in his Discord channel in plain English: "how did today go?", "what did you learn
last night?", "what are the QQQ levels?", "did everything run?". He answers from his real data
with a free AI model that runs on this PC (Ollama), and he posts a short report after the
9:35 AM trade, after the 6 PM study term, and if a job fails or didn't run.

He can only talk and report. Trading, pausing, changing tier and studying stay in the web app
and the automatic schedule.

One-time setup:

1. Ollama: install it from https://ollama.com and run `ollama pull qwen3:14b` (about 9.3 GB).
2. Discord Developer Portal (https://discord.com/developers/applications): **New Application**
   "Stan" > **Bot** > **Reset Token** > copy it. On the same page, turn on **Message Content Intent**.
3. **OAuth2** > **URL Generator**: tick **bot**, then **View Channels**, **Send Messages** and **Read
   Message History**. Open the link and add Stan to your private server.
4. In Discord: **User Settings** > **Advanced** > **Developer Mode** on. Right-click your name >
   **Copy User ID**; right-click Stan's channel > **Copy Channel ID**.
5. Put the three values in `.env`: `DISCORD_BOT_TOKEN`, `DISCORD_OWNER_ID`, `DISCORD_CHANNEL_ID`.
6. Double-click `schedule_trading.bat` once more. It adds the **Stan Discord** task, which starts
   the bot now and whenever you log in.

- Stan answers only you, only in that channel.
- Ask a question from a terminal without Discord: `python -m agent.discord_bot.brain "how did today go?"`
- Faster but less careful brain: set `STAN_MODEL=qwen3:8b` in `.env`.
- The bot's log is `data/discord.log`. To stop it for good: `schtasks /Delete /TN "Stan Discord" /F`
```

- [ ] **Step 5: Check the brain against real data**

Ask each question below. Compare every number in Stan's answer with the matching tool's `facts`, for example `python -c "from agent.discord_bot import tools as T; print(T.run_tool(T.ToolContext(), 'account', {}))"`, or with the web app (`stan.bat`):

```bash
python -m agent.discord_bot.brain "How much cash do I have?"
python -m agent.discord_bot.brain "What did you buy today?"
python -m agent.discord_bot.brain "How did school go last night?"
python -m agent.discord_bot.brain "Should I put real money in you yet?"
python -m agent.discord_bot.brain "What are the QQQ levels?"
python -m agent.discord_bot.brain "Did everything run today?"
python -m agent.discord_bot.brain "Pause trading."
```

Expected: every number matches its tool's `facts`. "Pause trading" gets a reply saying Stan can't do that from Discord and pointing to the web app. If any answer misstates a fact, fix the tool's fact wording or its description before going on. Don't change the rules to hide a mistake.

- [ ] **Step 6: Check `notify` against the real channel**

Run: `python -m agent.discord_bot.notify study 0`
Expected: prints `Sent the study report to Discord (1 message).`, and the report appears in Stan's channel with tonight's real numbers.

- [ ] **Step 7: Register and start the bot**

Run (PowerShell): `powershell -NoProfile -ExecutionPolicy Bypass -File .\schedule_discord.ps1`
Then: `Get-ScheduledTask -TaskName "Stan Discord" | Select-Object TaskName, State` and `(Get-ScheduledTask -TaskName "Stan Discord").Settings.ExecutionTimeLimit`
Expected: State `Running`; ExecutionTimeLimit `PT0S`. `data\discord.log` contains "Stan is online as Stan#..., listening in channel ...". If registering fails with "Access is denied", run the same command from PowerShell opened with **Run as administrator**.

- [ ] **Step 8: Live chat check (Zy)**

Zy writes "how much cash do I have?" in Stan's channel.
Expected: "Stan is typing..." appears, and the reply matches the account. A message from another account, or in another channel, gets no reply.

- [ ] **Step 9: Run the whole suite and commit**

Run: `python -m pytest -q`
Expected: 208 passed.

```bash
git add run_discord.bat schedule_discord.ps1 schedule_trading.bat README.md
git commit -m "Discord chat: keep the bot running, setup docs"
```
