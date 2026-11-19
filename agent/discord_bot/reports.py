"""Stan's Discord reports. Fixed wording built from the numbers, never from the AI model, so a
report can't misstate a number and still arrives when Ollama is off."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from agent.discord_bot.tools import ToolContext, last_log_entry, money, nice_day, run_tool
from agent.stan.paper import PaperBook

JOB_NAMES = {"open": "the 9:35 AM trading run", "study": "the 6 PM study term", "collect": "the 0DTE data collector",
             "box": "the 9:31 AM box breakout watch", "flatten": "the 3:55 PM box flatten"}
LOG_FILES = {"open": "open.log", "study": "study.log", "collect": "collect.log", "box": "box.log",
             "flatten": "flatten.log"}


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
            lines += [t.replace(" (today)", "") for t in todays]   # the header already says the date
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


def _box_state(ctx: ToolContext) -> dict | None:
    """Today's box breakout state from paper meta (read only)."""
    book = PaperBook(ctx.journal_path)
    try:
        state = book.get("box_today")
    finally:
        book.close()
    return state if state and state.get("day") == ctx.now().date().isoformat() else None


def _box(ctx: ToolContext) -> str:
    lines = [f"Box breakout for {nice_day(ctx.now().date())}:"]
    state = _box_state(ctx)
    if state is None:
        return "\n".join(lines + ["No record of today's watch in the paper log."])
    if state.get("box"):
        hi, lo = state["box"]
        lines.append(f"Box (9:00-9:30): high ${hi:.2f}, low ${lo:.2f}.")
    order = state.get("order") or {}
    if state.get("status") in ("sent", "closed", "flatten_failed") and order:
        verb = "buy" if order.get("side") == "buy" else "short"
        if order.get("stop") is not None:
            risk = (order.get("risk") or 0) * (order.get("qty") or 0)
            lines.append(f"Sent a bracket order to {verb} {order['qty']:g} QQQ at market: stop ${order['stop']:.2f}, "
                         f"target ${order['target']:.2f} (3R), risking about {money(risk)}.")
        else:
            lines.append(f"Today's bracket order {order.get('id')} is with Alpaca.")
    else:
        lines.append(f"No trade today: {state.get('reason') or state.get('status')}.")
    lines.append("Paper money only.")
    return "\n".join(lines)


def _flatten(ctx: ToolContext) -> str | None:
    state = _box_state(ctx)
    if state is None or not state.get("order"):
        return None                    # no box trade today, nothing to say
    day = nice_day(ctx.now().date())
    if state.get("status") == "flatten_failed":
        return (f"Problem: I couldn't close the box position for {day}: {state.get('reason')}. "
                "It may still be open; check the paper account.")
    res = state.get("result")
    if state.get("status") != "closed" or not res:
        return f"Box trade for {day}: still open after the flatten run; check the paper account."
    if not res.get("qty"):
        return f"Box trade for {day}: the entry order never filled, so there was nothing to close."
    how = {"time": "closed at 3:55", "stop": "stopped out", "target": "hit the 3R target"}.get(res["reason"], "closed")
    r = "" if res.get("r") is None else f"{res['r']:+.2f}R "
    return (f"Box trade for {day}: {'long' if res['side'] == 'buy' else 'short'} {res['qty']:g} QQQ from "
            f"${res['entry']:.2f}, {how} at ${res['exit']:.2f}: {r}({money(res.get('pnl'))}).\n"
            "Paper money only.")


def job_report(ctx: ToolContext, job: str, code: int) -> str | None:
    if job not in JOB_NAMES:
        raise ValueError(f"unknown job {job}")
    if code != 0:
        return _failure(job, code, ctx.data_dir)
    if job == "open":
        return _open(ctx)
    if job == "study":
        return _study(ctx)
    if job == "box":
        return _box(ctx)
    if job == "flatten":
        return _flatten(ctx)
    return None


def missed_run_message(job: str, data_dir: Path, today: date) -> str | None:
    entry = last_log_entry(Path(data_dir) / LOG_FILES[job])
    if entry is not None and entry.day == today:
        return None
    return (f"Heads up: {JOB_NAMES[job]} hasn't run today (nothing in data/{LOG_FILES[job]} for "
            f"{nice_day(today)}). The PC was probably off or asleep at that time.")
