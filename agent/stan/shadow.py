"""Stan's shadow trading: practice on live market days, with no orders.

Run once a day after the market closes (4:30pm ET or later):
    python -m agent.stan.shadow

Each run Stan does two jobs:

1. Selloff bounce, shadowed. He follows the fixed rule from his strategy exam
   (buy at the next open after RSI closes under 30, sell when RSI reaches 50 or
   after 10 days) on days that happen after shadowing starts, and compares the
   live results with what the backtest promised.

2. Beat-or-lag calls. To break the "always guess up" habit, Stan no longer calls
   whether stocks go up. Every day he must name the 5 stocks he expects to BEAT
   the market (SPY) tomorrow and the 5 he expects to LAG it. Because half of all
   stocks beat the market on any day, always saying "beat" scores about 50%, so
   there are no free points for optimism, and every day he has to make as many
   "lag" calls as "beat" calls.

Nothing here places a trade.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from agent.stan import daytrade
from agent.stan.persona import pct, signed_pct
from agent.stan.strategies import BASE_COST, BASE_RULE, SelloffRule
from agent.stan.study import FEATURES, make_features, rsi

ET = ZoneInfo("America/New_York")
PICKS_PER_SIDE = 5
MARKET = "SPY"
TRADES_NEEDED = 100  # rough number of closed trades before live results mean much

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shadow_positions (
    symbol TEXT NOT NULL, signal_date TEXT NOT NULL, status TEXT NOT NULL,
    entry_date TEXT, entry_price REAL, exit_date TEXT, exit_price REAL,
    last_price REAL, days INTEGER, net_ret REAL, first_seen TEXT NOT NULL,
    PRIMARY KEY (symbol, signal_date)
);
CREATE TABLE IF NOT EXISTS shadow_calls (
    made_on TEXT NOT NULL, symbol TEXT NOT NULL, call TEXT NOT NULL, p_beat REAL NOT NULL,
    graded INTEGER NOT NULL DEFAULT 0, graded_on TEXT, stock_ret REAL, market_ret REAL,
    correct INTEGER, PRIMARY KEY (made_on, symbol)
);
CREATE TABLE IF NOT EXISTS shadow_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT NOT NULL,
    kind TEXT NOT NULL, text TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(ET).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------- storage
class ShadowBook:
    """Shadow records live in Stan's journal file, in tables his school never wipes."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def get(self, key: str, default=None):
        row = self._db.execute("SELECT value FROM shadow_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self._db.execute("INSERT OR REPLACE INTO shadow_meta VALUES (?, ?)", (key, json.dumps(value)))
        self._db.commit()

    def log(self, day: date, kind: str, text: str) -> None:
        self._db.execute("INSERT INTO shadow_log (ts, day, kind, text) VALUES (?, ?, ?, ?)",
                         (now_iso(), day.isoformat(), kind, text))
        self._db.commit()

    def position(self, symbol: str, signal_date: str) -> dict | None:
        row = self._db.execute("SELECT * FROM shadow_positions WHERE symbol=? AND signal_date=?",
                               (symbol, signal_date)).fetchone()
        return dict(row) if row else None

    def save_position(self, p: dict) -> None:
        old = self.position(p["symbol"], p["signal_date"])
        if old and old["status"] == "closed":
            return  # a closed trade is final; later data revisions never rewrite it
        self._db.execute(
            "INSERT OR REPLACE INTO shadow_positions VALUES "
            "(:symbol, :signal_date, :status, :entry_date, :entry_price, :exit_date, :exit_price, "
            ":last_price, :days, :net_ret, :first_seen)",
            {**p, "first_seen": old["first_seen"] if old else now_iso()},
        )
        self._db.commit()

    def positions(self) -> list[dict]:
        return [dict(r) for r in self._db.execute(
            "SELECT * FROM shadow_positions ORDER BY signal_date DESC, symbol")]

    def add_calls(self, made_on: date, calls: list[dict]) -> int:
        added = 0
        for c in calls:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO shadow_calls (made_on, symbol, call, p_beat) VALUES (?, ?, ?, ?)",
                (made_on.isoformat(), c["symbol"], c["call"], c["p_beat"]))
            added += cur.rowcount
        self._db.commit()
        return added

    def ungraded_calls(self) -> list[dict]:
        return [dict(r) for r in self._db.execute("SELECT * FROM shadow_calls WHERE graded=0")]

    def grade_call(self, made_on: str, symbol: str, graded_on: str, stock_ret: float,
                   market_ret: float, correct: bool) -> None:
        self._db.execute(
            "UPDATE shadow_calls SET graded=1, graded_on=?, stock_ret=?, market_ret=?, correct=? "
            "WHERE made_on=? AND symbol=?",
            (graded_on, stock_ret, market_ret, int(correct), made_on, symbol))
        self._db.commit()

    def calls(self) -> list[dict]:
        return [dict(r) for r in self._db.execute(
            "SELECT * FROM shadow_calls ORDER BY made_on DESC, call, p_beat DESC")]

    def logs(self, limit: int = 200) -> list[dict]:
        rows = self._db.execute("SELECT * FROM shadow_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- data helpers
def complete_bars(history: dict[str, pd.DataFrame], now: datetime) -> dict[str, pd.DataFrame]:
    """Drop today's bar until 4:30pm ET, so Stan never acts on an unfinished day."""
    now = now.astimezone(ET)
    cutoff = now.date() if now.time() >= time(16, 30) else now.date() - timedelta(days=1)
    out = {}
    for symbol, df in history.items():
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[symbol] = df[df.index.date <= cutoff]
    return out


def latest_day(history: dict[str, pd.DataFrame]) -> date:
    market = history.get(MARKET)
    frame = market if market is not None and len(market) else next(iter(history.values()))
    return frame.index.max().date()


# ---------------------------------------------------------------- selloff shadow
def rule_positions(bars: pd.DataFrame, symbol: str, since: date,
                   rule: SelloffRule = BASE_RULE, cost: float = BASE_COST) -> list[dict]:
    """The rule's trades for signals on or after `since`, including ones still open.

    status: 'pending' (signal on the last day, buy at the next open), 'open' or 'closed'.
    """
    bars = bars.sort_index().dropna(subset=["open", "close"])
    c = bars["close"].astype(float).to_numpy()
    o = bars["open"].astype(float).to_numpy()
    dates = bars.index
    r = rsi(bars["close"].astype(float)).to_numpy()
    n = len(c)
    start = int(np.searchsorted(dates.date, since))
    out, i = [], start
    while i < n:
        if np.isnan(r[i]) or r[i] >= rule.entry_rsi:
            i += 1
            continue
        base = {"symbol": symbol, "signal_date": dates[i].date().isoformat(),
                "entry_date": None, "entry_price": None, "exit_date": None, "exit_price": None,
                "last_price": float(c[i]), "days": 0, "net_ret": None}
        if i == n - 1:
            out.append({**base, "status": "pending"})
            break
        entry = float(o[i + 1])
        j = i + 1
        while True:
            if (not np.isnan(r[j]) and r[j] >= rule.exit_rsi) or (j - i) >= rule.max_days:
                status = "closed"
                break
            if j == n - 1:
                status = "open"
                break
            j += 1
        gross = float(c[j] / entry - 1)
        out.append({**base, "status": status, "entry_date": dates[i + 1].date().isoformat(),
                    "entry_price": entry, "last_price": float(c[j]), "days": int(j - i),
                    "exit_date": dates[j].date().isoformat() if status == "closed" else None,
                    "exit_price": float(c[j]) if status == "closed" else None,
                    # Open trades show their mark-to-market return, including both costs.
                    "net_ret": gross - 2 * cost})
        if status == "open":
            break
        i = j + 1
    return out


def selloff_stats(positions: list[dict]) -> dict:
    closed = [p for p in positions if p["status"] == "closed"]
    rets = np.array([p["net_ret"] for p in closed], dtype=float)
    wins, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "open": sum(p["status"] == "open" for p in positions),
        "pending": sum(p["status"] == "pending" for p in positions),
        "closed": int(len(rets)),
        "win_rate": round(float((rets > 0).mean()), 4) if len(rets) else None,
        "avg": round(float(rets.mean()), 5) if len(rets) else None,
        "profit_factor": (round(float(wins / losses), 3) if losses > 0 else (99.0 if len(rets) else None)),
        "needed": TRADES_NEEDED,
    }


# ---------------------------------------------------------------- beat-or-lag calls
def relative_training_set(history: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows with a known next-day result, the latest row per stock to make calls on)."""
    if MARKET not in history:
        raise ValueError(f"{MARKET} is needed as the market benchmark")
    mkt = history[MARKET].sort_index()["close"].astype(float)
    mkt_next = (mkt.shift(-1) / mkt - 1).rename("market_next")
    frames = []
    for symbol, bars in history.items():
        if symbol == MARKET:
            continue
        bars = bars.sort_index()
        bars = bars[~bars.index.duplicated(keep="last")]
        f = make_features(bars).join(mkt_next, how="left")
        f["symbol"] = symbol
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    ready = panel.dropna(subset=FEATURES)
    last_date = ready["date"].max()
    latest = ready[ready["date"] == last_date]
    train = ready.dropna(subset=["next_ret", "market_next"]).copy()
    train["beat"] = (train["next_ret"] > train["market_next"]).astype(int)
    return train, latest


def make_calls(history: dict[str, pd.DataFrame], picks: int = PICKS_PER_SIDE,
               max_iter: int = 150, train_years: int | None = None) -> tuple[date, list[dict]]:
    """Train on every finished day, then name `picks` stocks to beat and `picks` to lag tomorrow."""
    train, latest = relative_training_set(history)
    if train_years:
        train = train[train["date"] >= train["date"].max() - pd.Timedelta(days=365 * train_years)]
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=max_iter, learning_rate=0.05, random_state=7)
    model.fit(train[FEATURES], train["beat"])
    latest = latest.copy()
    latest["p_beat"] = model.predict_proba(latest[FEATURES])[:, 1]
    ranked = latest.sort_values("p_beat", ascending=False)
    k = min(picks, len(ranked) // 2)
    calls = ([{"symbol": r.symbol, "call": "beat", "p_beat": round(float(r.p_beat), 4)} for r in ranked.head(k).itertuples()]
             + [{"symbol": r.symbol, "call": "lag", "p_beat": round(float(r.p_beat), 4)} for r in ranked.tail(k).itertuples()])
    return latest["date"].max().date(), calls


def grade_calls(book: ShadowBook, history: dict[str, pd.DataFrame]) -> int:
    """Grade every call whose next trading day has finished."""
    mkt = history[MARKET]["close"].astype(float)
    graded = 0
    for call in book.ungraded_calls():
        made_on = pd.Timestamp(call["made_on"])
        bars = history.get(call["symbol"])
        if bars is None or made_on not in bars.index or made_on not in mkt.index:
            continue
        closes = bars["close"].astype(float)
        after = closes.index[closes.index > made_on]
        if not len(after) or after[0] not in mkt.index:
            continue
        nxt = after[0]
        stock_ret = float(closes[nxt] / closes[made_on] - 1)
        market_ret = float(mkt[nxt] / mkt[made_on] - 1)
        rel = stock_ret - market_ret
        correct = rel > 0 if call["call"] == "beat" else rel < 0
        book.grade_call(call["made_on"], call["symbol"], nxt.date().isoformat(), stock_ret, market_ret, correct)
        graded += 1
    return graded


def call_stats(calls: list[dict], window_days: int = 20) -> dict:
    graded = [c for c in calls if c["graded"]]
    beats = [c for c in graded if c["call"] == "beat"]
    lags = [c for c in graded if c["call"] == "lag"]

    def hit(rows):
        return round(float(np.mean([r["correct"] for r in rows])), 4) if rows else None

    spreads = []
    for day in sorted({c["made_on"] for c in graded}):
        b = [c["stock_ret"] - c["market_ret"] for c in beats if c["made_on"] == day]
        l = [c["stock_ret"] - c["market_ret"] for c in lags if c["made_on"] == day]
        if b and l:
            spreads.append(float(np.mean(b) - np.mean(l)))
    recent_days = sorted({c["made_on"] for c in calls})[-window_days:]
    recent = [c for c in calls if c["made_on"] in recent_days]
    beat_share = (sum(c["call"] == "beat" for c in recent) / len(recent)) if recent else None
    # What "always say beat" would have scored on the same stocks: the habit Stan is breaking.
    always_beat = (round(float(np.mean([(c["stock_ret"] - c["market_ret"]) > 0 for c in graded])), 4)
                   if graded else None)
    return {
        "graded": len(graded),
        "days_graded": len(spreads),
        "hit_rate": hit(graded),
        "beat_hit_rate": hit(beats),
        "lag_hit_rate": hit(lags),
        "balanced_hit_rate": (round((hit(beats) + hit(lags)) / 2, 4) if beats and lags else None),
        "always_beat_rate": always_beat,
        "avg_daily_spread": round(float(np.mean(spreads)), 5) if spreads else None,
        "total_spread": round(float(np.prod([1 + s for s in spreads]) - 1), 5) if spreads else None,
        "beat_share_recent": round(beat_share, 4) if beat_share is not None else None,
    }


# ---------------------------------------------------------------- one daily run
def run_day(book: ShadowBook, history: dict[str, pd.DataFrame], now: datetime | None = None,
            demo: bool = False, max_iter: int = 150, train_years: int | None = None,
            expected: dict | None = None) -> dict:
    now = now or datetime.now(ET)
    history = complete_bars(history, now)
    day = latest_day(history)
    started = book.get("started")
    if started is None:
        # Shadowing starts with the first finished day after this run. Signals on or
        # before today were visible when the rule was written, so they don't count.
        started = (day + timedelta(days=1)).isoformat()
        book.set("started", started)
        book.set("rule", BASE_RULE.describe())
        book.set("cost", BASE_COST)
        book.log(day, "milestone", f"Day one of shadow trading. From tomorrow I follow the selloff bounce live "
                                   f"({BASE_RULE.describe()}) and make 5 beat and 5 lag calls every day. No real orders.")
    book.set("mode", "demo" if demo else "real")
    if expected is not None:
        book.set("expected", expected)
    last_done = book.get("last_day")

    # 1. grade yesterday's calls
    newly_graded = grade_calls(book, history)
    stats = call_stats(book.calls())
    if newly_graded:
        today_rows = [c for c in book.calls() if c["graded"] and c["graded_on"] == day.isoformat()]
        if today_rows:
            right = sum(c["correct"] for c in today_rows)
            book.log(day, "win" if right * 2 > len(today_rows) else "fail",
                     f"Graded {len(today_rows)} calls from yesterday: {right} right. "
                     f"All-time: {pct(stats['hit_rate'])} right on {stats['graded']} calls "
                     f"(beat calls {pct(stats['beat_hit_rate'])}, lag calls {pct(stats['lag_hit_rate'])}).")

    # 2. selloff bounce
    since = date.fromisoformat(started)
    before = {(p["symbol"], p["signal_date"]): p["status"] for p in book.positions()}
    for symbol, bars in history.items():
        for p in rule_positions(bars, symbol, since):
            book.save_position(p)
    positions = book.positions()
    for p in positions:
        old = before.get((p["symbol"], p["signal_date"]))
        if p["status"] == "pending" and old is None:
            book.log(day, "thought", f"{p['symbol']} closed with RSI under 30. I'll buy it at tomorrow's open "
                                     f"(shadow only).")
        elif p["status"] == "open" and old in (None, "pending"):
            book.log(day, "progress", f"Bought {p['symbol']} at the open for ${p['entry_price']:.2f} (shadow).")
        elif p["status"] == "closed" and old != "closed":
            book.log(day, "win" if p["net_ret"] > 0 else "fail",
                     f"Sold {p['symbol']} at ${p['exit_price']:.2f} after {p['days']} day(s): "
                     f"{signed_pct(p['net_ret'], 2)} after costs.")
    sell = selloff_stats(positions)

    # 3. tomorrow's beat-or-lag calls
    made_on, calls = make_calls(history, max_iter=max_iter, train_years=train_years)
    added = book.add_calls(made_on, calls)
    if added:
        beat = ", ".join(c["symbol"] for c in calls if c["call"] == "beat")
        lag = ", ".join(c["symbol"] for c in calls if c["call"] == "lag")
        book.log(day, "thought", f"Tomorrow I expect {beat} to beat the market and {lag} to lag it.")

    # 4. daily wrap-up
    if last_done != day.isoformat():
        msg = (f"Shadow day wrap-up: {sell['closed']} closed selloff trades so far"
               + (f", won {pct(sell['win_rate'])}, average {signed_pct(sell['avg'], 2)}" if sell["closed"] else "")
               + f"; {sell['open']} open. ")
        if expected and sell["closed"]:
            msg += (f"The backtest promised {pct(expected.get('win_rate'))} wins and "
                    f"{signed_pct(expected.get('avg'), 2)} per trade. ")
        if sell["closed"] < TRADES_NEEDED:
            msg += f"Too early to judge: I want about {TRADES_NEEDED} closed trades."
        book.log(day, "progress", msg)
        share = stats.get("beat_share_recent")
        if stats["graded"] and stats["always_beat_rate"] is not None:
            ahead = (stats["hit_rate"] or 0) > stats["always_beat_rate"]
            book.log(day, "win" if ahead else "lesson",
                     f"Habit check: saying 'beat' every time would have been right {pct(stats['always_beat_rate'])} "
                     f"of the time. I'm at {pct(stats['hit_rate'])}, with {pct(share)} of my recent calls on the "
                     "'beat' side. " + ("My two-sided calls are ahead of the habit." if ahead else
                     "I'm behind the habit right now, so my calls still need work, but I won't fall back on it."))
    book.set("last_day", day.isoformat())
    book.set("last_run", now_iso())
    return {"day": day.isoformat(), "selloff": sell, "calls": call_stats(book.calls()), "made_on": made_on.isoformat()}


def snapshot(book: ShadowBook) -> dict:
    """Everything the app's Shadow tab needs."""
    positions = book.positions()
    calls = book.calls()
    started = book.get("started")
    latest_made = calls[0]["made_on"] if calls else None
    return {
        "started": started,
        "last_day": book.get("last_day"),
        "last_run": book.get("last_run"),
        "mode": book.get("mode", "none"),
        "rule": book.get("rule"),
        "expected": book.get("expected"),
        "selloff": {
            "stats": selloff_stats(positions),
            "open": [p for p in positions if p["status"] in ("open", "pending")],
            "recent_closed": sorted((p for p in positions if p["status"] == "closed"),
                                    key=lambda p: (p["exit_date"] or "", p["symbol"]), reverse=True)[:20],
        },
        "calls": {
            "stats": call_stats(calls),
            "latest_day": latest_made,
            "latest": [c for c in calls if c["made_on"] == latest_made],
        },
        "practice": _practice_snapshot(book.path),
        "log": book.logs(100),
    }


def _practice_snapshot(path: str) -> dict:
    pb = daytrade.PracticeBook(path)
    try:
        rows = pb.rows()
    finally:
        pb.close()
    return {"stats": daytrade.practice_stats(rows), "recent": rows[:40]}


def run_practice(book: ShadowBook, day: date, symbols: Sequence[str], demo: bool = False,
                 replays: int = 10000) -> list[dict]:
    """The evening's day-trade practice, walked with the Day Trader skill pack."""
    pb = daytrade.PracticeBook(book.path)
    try:
        if pb.has_day(day.isoformat()):
            return []
        texts = daytrade.load_charts([s for s in symbols if s != MARKET], demo=demo,
                                     log=lambda m: book.log(day, "thought", f"5-minute data missing: {m}"))
        return daytrade.practice_day(pb, texts, day.isoformat(), replays=replays,
                                     log=lambda kind, text: book.log(day, kind, text))
    finally:
        pb.close()


# ---------------------------------------------------------------- command line
def expected_from_school(journal_path: str | Path) -> dict | None:
    """The backtest numbers the live results get compared with, from Stan's last school run."""
    try:
        with sqlite3.connect(str(journal_path)) as db:
            row = db.execute("SELECT value FROM meta WHERE key='strategy'").fetchone()
        if row:
            score = json.loads(row[0])["score"]
            return {"win_rate": score.get("win_rate"), "avg": score.get("avg"),
                    "profit_factor": score.get("profit_factor"), "trades": score.get("trades")}
    except (sqlite3.Error, KeyError, ValueError):
        pass
    return None


def main(argv: Sequence[str] | None = None) -> int:
    from agent.stan.train import DEFAULT_SYMBOLS, demo_history, real_history

    parser = argparse.ArgumentParser(description="Run one day of Stan's shadow trading.")
    parser.add_argument("--journal", default="data/stan.db")
    parser.add_argument("--db", default="data/agent.db", help="Alpaca bars database (fallback source)")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--demo", action="store_true",
                        help="replay the last 40 days of made-up practice data, to try the Shadow tab")
    args = parser.parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if MARKET not in symbols:
        symbols.insert(0, MARKET)

    if args.demo:
        demo_path = str(Path(args.journal).with_name("stan_shadow_demo.db"))
        Path(demo_path).unlink(missing_ok=True)
        book = ShadowBook(demo_path)
        full = demo_history(symbols, years=6)
        days = full[MARKET].index
        for k in range(len(days) - 40, len(days) + 1):
            cut = {s: df[df.index <= days[k - 1]] for s, df in full.items()}
            when = datetime.combine(days[k - 1].date(), time(17, 0), tzinfo=ET)
            run_day(book, cut, now=when, demo=True, max_iter=40, train_years=2,
                    expected={"win_rate": 0.553, "avg": 0.0002, "profit_factor": 1.009, "trades": 3419})
        pb = daytrade.PracticeBook(demo_path)
        texts = daytrade.load_charts([s for s in symbols if s != MARKET][:15], demo=True)
        for d in days[-8:]:
            daytrade.practice_day(pb, texts, d.date().isoformat(), replays=2000,
                                  log=lambda kind, text, d=d: book.log(d.date(), kind, text))
        pb.close()
        print(f"Practice replay saved to {demo_path}. Start the server with "
              f"STAN_JOURNAL={demo_path} to view it, or just open the Shadow tab in demo mode.")
        book.close()
        return 0

    history, source = real_history(symbols, 15, args.db)
    if MARKET not in history:
        print(f"Could not load {MARKET} prices from {source}; nothing to do.")
        return 1
    book = ShadowBook(args.journal)
    result = run_day(book, history, expected=expected_from_school(args.journal))
    practiced = run_practice(book, date.fromisoformat(result["day"]), symbols)
    s, c = result["selloff"], result["calls"]
    print(f"Shadow day {result['day']} ({source}).")
    print(f"  Selloff bounce: {s['closed']} closed, {s['open']} open, {s['pending']} buying at the next open."
          + (f" Won {pct(s['win_rate'])}, average {signed_pct(s['avg'], 2)}." if s["closed"] else ""))
    print(f"  Beat-or-lag calls: {c['graded']} graded, {pct(c['hit_rate'])} right "
          f"(beat {pct(c['beat_hit_rate'])}, lag {pct(c['lag_hit_rate'])}).")
    print(f"  New calls made for the day after {result['made_on']}.")
    taken = sum(r["taken"] for r in practiced)
    print(f"  Day-trade practice: {len(practiced)} opening-range breakouts walked through the skill pack, "
          f"{taken} would have been taken on the $50 plan.")
    book.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
