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

from agent.stan.journal import Journal
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


def _relative(d: date, today: date) -> str:
    """' (today)' or ' (yesterday)' after a day, so the model can't call an older day 'today'."""
    return {0: " (today)", 1: " (yesterday)"}.get((today - d).days, "")


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
        on = f"On {nice_day(d)}{_relative(d, ctx.now().date())}" if d else "On an unknown day"
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
RESULT_WORDS = {0: "OK", 267009: "still running", 267011: "has not run yet",
                0xC000013A: "stopped: the window was closed or Ctrl+C"}


def _result_word(code) -> str:
    if code in RESULT_WORDS:
        return RESULT_WORDS[code]
    if isinstance(code, int) and (code < 0 or code > 0x7FFFFFFF):
        return f"error code 0x{code & 0xFFFFFFFF:08X}"
    return f"error code {code}"


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
               f"({_result_word(code & 0xFFFFFFFF if isinstance(code, int) and code < 0 else code)})") if last else "has never run"
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
         _params(day={"type": "string", "description": "A date like 2026-01-15. Leave it out for the latest saved day."}), levels),
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
    try:
        given = args if isinstance(args, dict) else {}
        kwargs = {k: v for k, v in given.items() if k in allowed and v not in (None, "")}
        return tool.func(ctx, **kwargs)
    except Exception as err:  # one bad data source must not break the whole answer
        return {"error": f"{type(err).__name__}: {err}"}
