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
