"""Stan's paper trading on the Alpaca paper account: the selloff bounce, with real (paper) orders.

Run once a day after the close (4:30pm ET or later), after shadow trading:
    python -m agent.stan.paper            # sync fills, exits, entries
    python -m agent.stan.paper --dry-run  # show what he would do, send nothing
    python -m agent.stan.paper --status   # print the account and positions
    python -m agent.stan.paper --resume   # restart after the kill switch (your call only)

The rule is the one from his strategy exam: when a stock's 14-day RSI closes
under 30, buy at the next open; sell at the next open after RSI closes at 50 or
higher, or once 10 trading days have passed since the signal. Orders are
market-on-open (OPG), so they fill in the opening auction like the backtest.

Paper only. The broker adapter refuses live settings, and so does this module.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from agent.broker.base import Broker
from agent.stan import tiers
from agent.stan.shadow import ET, MARKET, complete_bars, latest_day, now_iso
from agent.stan.strategies import BASE_RULE, SelloffRule
from agent.stan.study import rsi

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS paper_trades (
    symbol TEXT NOT NULL, signal_date TEXT NOT NULL, status TEXT NOT NULL,
    qty INTEGER NOT NULL, entry_order TEXT NOT NULL, entry_date TEXT, entry_price REAL,
    exit_signal_date TEXT, exit_reason TEXT, exit_order TEXT, exit_date TEXT, exit_price REAL,
    pnl REAL, ret REAL, created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, signal_date)
);
CREATE TABLE IF NOT EXISTS paper_equity (day TEXT PRIMARY KEY, equity REAL NOT NULL, cash REAL NOT NULL);
CREATE TABLE IF NOT EXISTS paper_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL
);
"""
# Statuses: pending_entry -> open -> pending_exit -> closed; or cancelled.
DONE = {"filled"}
DEAD = {"canceled", "expired", "rejected", "suspended", "stopped"}


@dataclass(frozen=True)
class PaperConfig:
    position_pct: float = 0.10       # each trade gets 10% of the account
    max_positions: int = 10
    kill_switch_drawdown: float = 0.20
    min_price: float = 5.0
    rule: SelloffRule = BASE_RULE


class PaperBook:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def get(self, key: str, default=None):
        row = self._db.execute("SELECT value FROM paper_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self._db.execute("INSERT OR REPLACE INTO paper_meta VALUES (?, ?)", (key, json.dumps(value)))
        self._db.commit()

    def log(self, day: str, kind: str, text: str) -> None:
        self._db.execute("INSERT INTO paper_log (ts, day, kind, text) VALUES (?, ?, ?, ?)",
                         (now_iso(), day, kind, text))
        self._db.commit()

    def logs(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self._db.execute("SELECT * FROM paper_log ORDER BY id DESC LIMIT ?", (limit,))]

    def trades(self, statuses: Sequence[str] | None = None) -> list[dict]:
        rows = [dict(r) for r in self._db.execute("SELECT * FROM paper_trades ORDER BY signal_date DESC, symbol")]
        return [r for r in rows if statuses is None or r["status"] in statuses]

    def add_trade(self, t: dict) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO paper_trades (symbol, signal_date, status, qty, entry_order, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (t["symbol"], t["signal_date"], t["status"], t["qty"], t["entry_order"], now_iso()))
        self._db.commit()

    def update(self, symbol: str, signal_date: str, **fields) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        self._db.execute(f"UPDATE paper_trades SET {sets} WHERE symbol=? AND signal_date=?",
                         [*fields.values(), symbol, signal_date])
        self._db.commit()

    def record_equity(self, day: str, equity: float, cash: float) -> None:
        self._db.execute("INSERT OR REPLACE INTO paper_equity VALUES (?, ?, ?)", (day, equity, cash))
        self._db.commit()

    def equity_history(self) -> list[dict]:
        return [dict(r) for r in self._db.execute("SELECT * FROM paper_equity ORDER BY day")]


def order_id(symbol: str, signal_date: str, side: str) -> str:
    """The same trade always gets the same id, so Alpaca refuses a duplicate."""
    return f"stan-{symbol}-{signal_date.replace('-', '')}-{side}"


def safe_submit(broker: Broker, symbol: str, qty: int, side: str, oid: str) -> tuple[bool, str]:
    """Submit once. If the call errors, check whether Alpaca took it anyway (same id)."""
    try:
        broker.submit_open_order(symbol, qty, side, oid)
        return True, ""
    except Exception as err:
        try:
            if broker.get_order(oid) is not None:
                return True, ""
        except Exception:
            pass
        return False, f"{type(err).__name__}: {err}"


# ---------------------------------------------------------------- the rule on today's close
def signals_today(history: dict[str, pd.DataFrame], day: date, rule: SelloffRule) -> list[dict]:
    out = []
    for sym, df in history.items():
        if sym == MARKET or df.empty or df.index.max().date() != day:
            continue
        r = rsi(df["close"].astype(float))
        val = r.iloc[-1]
        if not np.isnan(val) and val < rule.entry_rsi:
            out.append({"symbol": sym, "rsi": float(val), "close": float(df["close"].iloc[-1])})
    return sorted(out, key=lambda x: x["rsi"])  # most oversold first


def exit_due(df: pd.DataFrame, signal_date: str, day: date, rule: SelloffRule) -> str | None:
    """Same exit as the backtest: RSI closes at 50+, or 10 trading days since the signal."""
    idx = df.index
    if idx.max().date() != day:
        return None
    try:
        i = int(np.flatnonzero(idx.date == date.fromisoformat(signal_date))[0])
    except IndexError:
        return None
    j = len(df) - 1
    if j <= i:
        return None
    val = rsi(df["close"].astype(float)).iloc[-1]
    if not np.isnan(val) and val >= rule.exit_rsi:
        return f"RSI recovered to {val:.0f}"
    if j - i >= rule.max_days:
        return f"{rule.max_days} trading days passed"
    return None


# ---------------------------------------------------------------- one evening
def run_evening(broker: Broker, book: PaperBook, history: dict[str, pd.DataFrame],
                now: datetime | None = None, cfg: PaperConfig = PaperConfig(), dry_run: bool = False) -> dict:
    now = now or datetime.now(ET)
    history = complete_bars(history, now)
    day = latest_day(history)
    d = day.isoformat()
    actions: list[str] = []

    def say(kind: str, text: str) -> None:
        actions.append(text)
        if not dry_run:
            book.log(d, kind, text)

    account = broker.get_account()
    if not account.is_paper:
        raise RuntimeError("Stan only trades the paper account.")
    if book.get("start_equity") is None and not dry_run:
        book.set("start_equity", account.equity)
        book.set("started", d)
        strat = book.get("strategy", "selloff")
        rule = (f"the selloff bounce. Each trade gets {cfg.position_pct:.0%} of the account, up to "
                f"{cfg.max_positions} at once." if strat == "selloff"
                else "the box breakout on QQQ, risking 1% a trade." if strat == "box"
                else f"the {strat}-risk arena winner.")
        say("milestone", f"Paper trading starts with ${account.equity:,.2f} of paper money. Rule: {rule} "
                         f"Kill switch at {cfg.kill_switch_drawdown:.0%} below the start.")
    start = book.get("start_equity", account.equity)
    if not dry_run:
        book.record_equity(d, account.equity, account.cash)
    floor = start * (1 - cfg.kill_switch_drawdown)

    # 1. sync fills and dead orders
    for t in book.trades(["pending_entry"]):
        o = broker.get_order(t["entry_order"])
        if o is None:
            continue
        finished = o.status in DONE or o.status in DEAD
        if finished and o.filled_qty > 0 and not dry_run:
            book.update(t["symbol"], t["signal_date"], status="open", qty=int(o.filled_qty),
                        entry_price=o.filled_avg_price,
                        entry_date=(o.filled_at.astimezone(ET).date().isoformat() if o.filled_at else d))
            part = "" if o.filled_qty >= t["qty"] else f" (only {int(o.filled_qty)} of {t['qty']} filled)"
            say("progress", f"Bought {int(o.filled_qty)} {t['symbol']} at ${o.filled_avg_price:.2f} "
                            f"in the opening auction{part}.")
        elif o.status in DEAD and not dry_run:
            book.update(t["symbol"], t["signal_date"], status="cancelled")
            say("thought", f"The {t['symbol']} buy order was {o.status}; nothing was bought.")
    for t in book.trades(["pending_exit"]):
        o = broker.get_order(t["exit_order"])
        if o is None:
            continue
        if o.status in DONE and not dry_run:
            pnl = (o.filled_avg_price - t["entry_price"]) * t["qty"]
            ret = o.filled_avg_price / t["entry_price"] - 1
            book.update(t["symbol"], t["signal_date"], status="closed", exit_price=o.filled_avg_price,
                        exit_date=(o.filled_at.astimezone(ET).date().isoformat() if o.filled_at else d),
                        pnl=round(pnl, 2), ret=round(ret, 5))
            say("win" if pnl > 0 else "fail",
                f"Sold {t['qty']} {t['symbol']} at ${o.filled_avg_price:.2f}: {ret:+.2%} (${pnl:+,.2f}).")
        elif o.status in DEAD and not dry_run:
            left = t["qty"] - int(o.filled_qty)
            # Keep the exit reason: tonight's run sells whatever is left, no matter what RSI does now.
            book.update(t["symbol"], t["signal_date"], status="open", exit_order=None, qty=left)
            say("fail", f"The {t['symbol']} sell order was {o.status}"
                        + (f" after selling {int(o.filled_qty)}" if o.filled_qty else "")
                        + f". I'll sell the remaining {left} at the next open.")

    # 2. reconcile with Alpaca
    held = {p.symbol: p.qty for p in broker.get_positions()}
    from agent.stan.box import live as box_live   # late import: box.live imports this module
    box_qty = box_live.held_qty(broker, book)     # the box trade's QQQ isn't one of the selloff trades
    if box_qty:
        held[box_live.SYMBOL] = held.get(box_live.SYMBOL, 0) - box_qty
    mine: dict[str, float] = {}
    for t in book.trades(["open", "pending_exit"]):
        mine[t["symbol"]] = mine.get(t["symbol"], 0) + t["qty"]
    mismatched = [s for s, q in mine.items() if abs(held.get(s, 0) - q) > 1e-6]
    if mismatched and not dry_run:
        book.set("halted", True)
        book.set("halt_reason", "My records don't match Alpaca for: " + ", ".join(mismatched))
        say("fail", "Stopped: my records don't match Alpaca's positions for " + ", ".join(mismatched)
                    + ". Nothing new until you check and run --resume.")

    # 3. kill switch
    if account.equity <= floor and not book.get("halted", False):
        if not dry_run:
            broker.cancel_open_orders()
        tiers.sell_all(broker, d, say, dry_run)
        if box_qty:
            try:
                box_live.flatten(broker, book, now, dry_run=dry_run, log=lambda text: None)
            except Exception as err:
                say("fail", f"Couldn't close the box position: {type(err).__name__}: {err}")
        if not dry_run:
            for t in book.trades(["open"]):
                oid = order_id(t["symbol"], t["signal_date"], "sell" + d.replace("-", ""))
                ok, err = safe_submit(broker, t["symbol"], t["qty"], "sell", oid)
                if ok:
                    book.update(t["symbol"], t["signal_date"], status="pending_exit", exit_order=oid,
                                exit_signal_date=d, exit_reason="kill switch")
                else:
                    say("fail", f"Couldn't send the kill-switch sell for {t['symbol']}: {err}")
            book.set("halted", True)
            book.set("halt_reason", f"Kill switch: equity ${account.equity:,.2f} fell to {floor:,.2f} or below")
        say("fail", f"Kill switch: the account is at ${account.equity:,.2f}, at or below ${floor:,.2f} "
                    f"({cfg.kill_switch_drawdown:.0%} under the start). Selling everything at the next open and stopping.")

    if book.get("halted", False):
        if not dry_run:
            book.set("last_day", d)
        return {"day": d, "halted": True, "reason": book.get("halt_reason"), "actions": actions}

    # 4. exits
    for t in book.trades(["open"]):
        df = history.get(t["symbol"])
        why = t["exit_reason"] or (exit_due(df, t["signal_date"], day, cfg.rule) if df is not None else None)
        if why and t["qty"] > 0:
            oid = order_id(t["symbol"], t["signal_date"], "sell" + d.replace("-", ""))
            if not dry_run:
                ok, err = safe_submit(broker, t["symbol"], t["qty"], "sell", oid)
                if not ok:
                    say("fail", f"Couldn't send the {t['symbol']} sell order: {err}. I'll retry next run.")
                    continue
                book.update(t["symbol"], t["signal_date"], status="pending_exit", exit_order=oid,
                            exit_signal_date=d, exit_reason=why)
            say("thought", f"Selling {t['qty']} {t['symbol']} at tomorrow's open: {why}.")

    strategy = book.get("strategy", "selloff")
    open_selloff = book.trades(["pending_entry", "open", "pending_exit"])

    # 5a. the box breakout trades during the day (agent.stan.box.live); this run only clears the tiers
    if strategy == "box":
        if open_selloff:
            say("thought", f"On the box breakout: {len(open_selloff)} selloff trade(s) finish first. "
                           "No new selloff trades.")
        if not book.get("box_tiers_sold"):
            if any(p.symbol in tiers.TIER_SYMBOLS and p.qty > 0 for p in broker.get_positions()):
                tiers.sell_all(broker, d, say, dry_run)
            if not dry_run:
                book.set("box_tiers_sold", d)
        if not dry_run:
            book.set("last_day", d)
        return {"day": d, "halted": False, "equity": account.equity, "start": start, "floor": floor,
                "actions": actions, "strategy": strategy}

    # 5b. a tier strategy (once the selloff trades have closed)
    if strategy in ("low", "medium", "high"):
        if open_selloff:
            say("thought", f"Switching to the {strategy}-risk strategy: waiting for {len(open_selloff)} selloff "
                           "trade(s) to finish first. No new selloff trades.")
        else:
            prices = {s: float(df["close"].iloc[-1]) for s, df in history.items() if s in tiers.TIER_SYMBOLS}
            tiers.run_tier(broker, book, history, strategy, d, account.equity, prices, say, dry_run,
                           low_fraction=book.get("low_fraction", tiers.DEFAULT_LOW_FRACTION), cash=account.cash)
        if not dry_run:
            book.set("last_day", d)
        return {"day": d, "halted": False, "equity": account.equity, "start": start, "floor": floor,
                "actions": actions, "strategy": strategy}

    # 5c. selloff entries
    if book.get("last_entries_day") != d:
        active = book.trades(["pending_entry", "open", "pending_exit"])
        busy = {t["symbol"] for t in active}
        slots = cfg.max_positions - len(active)
        cash = account.cash
        for sig in signals_today(history, day, cfg.rule):
            if slots <= 0:
                say("thought", f"{sig['symbol']} signalled (RSI {sig['rsi']:.0f}) but all {cfg.max_positions} slots are full.")
                continue
            if sig["symbol"] in busy or sig["close"] < cfg.min_price:
                continue
            if held.get(sig["symbol"], 0) and sig["symbol"] not in mine:
                say("thought", f"{sig['symbol']} signalled, but the account already holds it outside my trades, "
                               "so I'm leaving it alone.")
                continue
            qty = int((account.equity * cfg.position_pct) // sig["close"])
            if qty <= 0 or qty * sig["close"] > cash:
                say("thought", f"{sig['symbol']} signalled but there isn't enough cash for a whole share.")
                continue
            oid = order_id(sig["symbol"], d, "buy")
            if not dry_run:
                ok, err = safe_submit(broker, sig["symbol"], qty, "buy", oid)
                if not ok:
                    say("fail", f"Couldn't send the {sig['symbol']} buy order: {err}")
                    continue
                book.add_trade({"symbol": sig["symbol"], "signal_date": d, "status": "pending_entry",
                                "qty": qty, "entry_order": oid})
            say("thought", f"Buying {qty} {sig['symbol']} at tomorrow's open (RSI closed at {sig['rsi']:.0f}, "
                           f"about ${qty * sig['close']:,.0f}).")
            slots -= 1
            cash -= qty * sig["close"]
            busy.add(sig["symbol"])
        if not dry_run:
            book.set("last_entries_day", d)
    if not dry_run:
        book.set("last_day", d)
    return {"day": d, "halted": False, "equity": account.equity, "start": start, "floor": floor, "actions": actions}


def snapshot(book: PaperBook, broker_positions: dict | None = None) -> dict:
    trades = book.trades()
    closed = [t for t in trades if t["status"] == "closed"]
    rets = np.array([t["ret"] for t in closed], dtype=float)
    eq = book.equity_history()
    start = book.get("start_equity")
    last = eq[-1]["equity"] if eq else None
    return {
        "enabled": start is not None,
        "approved_by": "Zy",
        "strategy": book.get("strategy", "selloff"),
        "tier_plan": book.get("tier_plan"),
        "started": book.get("started"),
        "start_equity": start,
        "equity": last,
        "change": (last / start - 1) if (start and last) else None,
        "kill_floor": start * (1 - PaperConfig().kill_switch_drawdown) if start else None,
        "halted": bool(book.get("halted", False)),
        "halt_reason": book.get("halt_reason"),
        "last_day": book.get("last_day"),
        "open": [t for t in trades if t["status"] in ("pending_entry", "open", "pending_exit")],
        "closed": closed[:30],
        "stats": {
            "closed": int(len(rets)),
            "win_rate": round(float((rets > 0).mean()), 4) if len(rets) else None,
            "avg": round(float(rets.mean()), 5) if len(rets) else None,
            "pnl": round(float(sum(t["pnl"] or 0 for t in closed)), 2),
        },
        "equity_history": eq[-120:],
        "log": book.logs(60),
    }


# ---------------------------------------------------------------- command line
def _low_fraction_from_arena(journal_path: str) -> float | None:
    """The size the last arena run chose for the low-risk tier, if it sized down volatility target."""
    try:
        with sqlite3.connect(journal_path) as db:
            row = db.execute("SELECT value FROM meta WHERE key='arena'").fetchone()
        if not row:
            return None
        arena = json.loads(row[0])
        key = arena["winners"].get("low")
        hit = next((x for x in arena["strategies"] if x.get("key") == key), None)
        if hit and hit.get("base") == "vol_target" and hit.get("fraction"):
            return float(hit["fraction"])
    except (sqlite3.Error, KeyError, ValueError, TypeError):
        pass
    return None


def set_strategy(book: PaperBook, journal_path: str, name: str) -> str:
    """Choose what the paper account trades from the next run on. Returns a one-line confirmation."""
    if name not in ("selloff", "low", "medium", "high", "box"):
        raise ValueError(f"unknown strategy {name}")
    frac = tiers.DEFAULT_LOW_FRACTION
    if name == "low":
        frac = _low_fraction_from_arena(str(journal_path)) or frac
        book.set("low_fraction", frac)
    old = book.get("strategy", "selloff")
    if old == name:
        return f"Paper account is already on: {name}."
    book.set("strategy", name)
    book.set("tier_orders_day", None)   # a new choice may trade the same evening
    if name == "box":
        book.set("box_tiers_sold", None)   # the next run sells the tier holdings, once
    book.log(datetime.now(ET).date().isoformat(), "milestone",
             f"Zy switched the paper account from {old} to {name}" + (f" ({frac:.0%} size)" if name == "low" else "") + ".")
    return (f"Paper account set to: {name}" + (f" (low risk at {frac:.0%} size)" if name == "low" else "")
            + ". It takes effect at the next run; open selloff trades finish first.")


def main(argv: Sequence[str] | None = None) -> int:
    from agent.broker.alpaca import AlpacaBroker
    from agent.config import ConfigError, load_settings
    from agent.stan.train import DEFAULT_SYMBOLS, real_history

    parser = argparse.ArgumentParser(description="Stan's paper trading on the Alpaca paper account.")
    parser.add_argument("--journal", default="data/stan.db")
    parser.add_argument("--db", default="data/agent.db")
    parser.add_argument("--dry-run", action="store_true", help="show what Stan would do; send no orders")
    parser.add_argument("--status", action="store_true", help="print the paper account and Stan's positions")
    parser.add_argument("--resume", action="store_true", help="restart after the kill switch or a mismatch")
    parser.add_argument("--use", choices=["selloff", "low", "medium", "high", "box"],
                        help="choose what the paper account trades from now on (your call)")
    args = parser.parse_args(argv)

    if args.use:
        book = PaperBook(args.journal)
        try:
            print(set_strategy(book, args.journal, args.use))
            return 0
        finally:
            book.close()

    try:
        settings = load_settings()
    except ConfigError as err:
        print(f"Setup problem: {err}")
        return 1
    broker = AlpacaBroker(settings)
    book = PaperBook(args.journal)
    try:
        if args.resume:
            book.set("halted", False)
            book.set("halt_reason", None)
            book.log(datetime.now(ET).date().isoformat(), "milestone", "Zy restarted paper trading.")
            print("Paper trading restarted. Stan will act at the next evening run.")
            return 0
        if args.status:
            a = broker.get_account()
            snap = snapshot(book)
            print(f"Paper account: ${a.equity:,.2f} equity, ${a.cash:,.2f} cash."
                  + (f" Started at ${snap['start_equity']:,.2f}; kill switch at ${snap['kill_floor']:,.2f}."
                     if snap["start_equity"] else " Stan hasn't started yet."))
            for t in snap["open"]:
                print(f"  {t['symbol']}: {t['qty']} shares, {t['status']}"
                      + (f", bought at ${t['entry_price']:.2f}" if t["entry_price"] else ""))
            print("  Halted: " + (snap["halt_reason"] or "yes") if snap["halted"] else "  Running.")
            return 0
        history, source = real_history(list(DEFAULT_SYMBOLS) + ["TQQQ"], 2, args.db)
        if MARKET not in history:
            print(f"Could not load prices from {source}; nothing done.")
            return 1
        result = run_evening(broker, book, history, dry_run=args.dry_run)
        print(f"Paper trading, {result['day']}" + (" (dry run, nothing sent)" if args.dry_run else "") + ":")
        for line in result["actions"] or ["Nothing to do tonight."]:
            print("  " + line)
        if result["halted"]:
            print(f"  Halted: {result['reason']}. Run --resume once you've checked.")
        return 0
    finally:
        book.close()


if __name__ == "__main__":
    sys.exit(main())
