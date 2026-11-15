"""Stan's day trading, built on the Day Trader skill pack (agent/skills/day_trader).

Two jobs:

1. The day-trading exam (school, term 3). The pack's opening-range breakout,
   run exactly the way its Time Machine replays history: on every full session
   in the last 60 days of free 5-minute bars, for every stock Stan watches. Then
   the same kind of stress tests as the selloff bounce.

2. Day-trade practice (every evening, with shadow trading). For each stock's
   finished session Stan walks the pack's chain for one trade: scanner rule
   (did it break its opening range?), entry timer (the first 5-minute close
   above), exit planner (stop at the range midpoint, 1R target, out by 3:55),
   Time Machine (10,000 replays plus the rule's own history, before knowing the
   outcome), then the budget manager and risk manager on the $50 plan. Only
   then does he look at what really happened and grade it.

The pack's code is loaded unchanged. Nothing here places an order: the pack's
order executor stays locked.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "day_trader"
ET = ZoneInfo("America/New_York")
OR_MINUTES = 15
OR_BARS = OR_MINUTES // 5
UNTIL = "15:55"                                   # flat by 3:55 PM, the pack's default
LAST_BAR = (15 * 60 + 55 - (9 * 60 + 30)) // 5 - 1  # index of the 3:50 bar (closes 3:55)
SESSION_BARS = 78
BASE_COST = 0.0005   # 0.05% per side: spread and slippage on liquid stocks, 5-minute bars
HIGH_COST = 0.0010
MIN_RISK_PCT = 0.0005  # the exit planner refuses a stop tighter than 0.05% of the price


# ---------------------------------------------------------------- loading the pack
@lru_cache(maxsize=None)
def skill(relpath: str):
    """Load one of the pack's tools as a module, unchanged."""
    path = SKILL_DIR / relpath
    spec = importlib.util.spec_from_file_location("daytrader_" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timemachine():
    return skill("trader-timemachine/timemachine.py")


def fetch_chart_text(symbol: str, offline: str | os.PathLike | None = None) -> str:
    """60 days of 5-minute bars from Yahoo, through the pack's own fetcher."""
    tm = timemachine()
    if offline:
        return tm.read_offline(str(offline), symbol)
    return tm.fetch_chart(symbol)


def sessions_from_text(text: str, symbol: str, now_utc: datetime) -> dict:
    """{date: {slot: bar}} of complete regular-session 5-minute bars."""
    tm = timemachine()
    _, bars, _ = tm.parse_bars(tm.chart_result(text, symbol), now_utc)
    return tm.split_sessions(bars)


def full_sessions(days: dict, before: str = "9999-12-31") -> list[tuple[str, list[dict]]]:
    return timemachine().full_sessions(days, before)[0]


# ---------------------------------------------------------------- one session, the pack's rule
def orb_trade(day: str, bars: list[dict], target_r: float = 1.0) -> dict | None:
    """The opening-range breakout on one full session, as the pack's history replay runs it."""
    tm = timemachine()
    oh, ol = tm.opening_range(bars, OR_BARS)
    k = next((i for i in range(OR_BARS, LAST_BAR + 1) if bars[i]["c"] > oh), None)
    if k is None:
        return None
    entry = round(bars[k]["c"], 2)
    stop = tm.cents_away((oh + ol) / 2, True)
    risk = entry - stop
    if risk <= 0:
        return None
    target = round(entry + target_r * risk, 2)
    outcome, r = _play_out(bars, k, entry, stop, target, risk)
    return {"date": day, "entry_bar": k, "entry_time": tm.clock12(9 * 60 + 30 + 5 * (k + 1)),
            "or_high": oh, "or_low": ol, "entry": entry, "stop": stop, "target": target,
            "risk": round(risk, 4), "target_r": target_r, "outcome": outcome, "R": round(r, 3)}


def _play_out(bars: list[dict], k: int, entry: float, stop: float, target: float, risk: float) -> tuple[str, float]:
    """Bar by bar after the entry: stop first, then target, else the close at 3:55."""
    for b in bars[k + 1:LAST_BAR + 1]:
        if b["l"] <= stop:
            return "stop", -1.0
        if b["h"] >= target:
            return "target", (target - entry) / risk
    return "close", (bars[LAST_BAR]["c"] - entry) / risk


def net_r(trade: dict, cost: float) -> float:
    """R after costs: both sides' cost as a share of the price, in units of risk."""
    return trade["R"] - 2 * cost * trade["entry"] / trade["risk"]


# ---------------------------------------------------------------- the exam
def _score(trades: list[dict], cost: float) -> dict:
    r = np.array([net_r(t, cost) for t in trades])
    if not len(r):
        return {"trades": 0, "win_rate": None, "profit_factor": None, "avg": None, "avg_R": None}
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": int(len(r)), "win_rate": round(float((r > 0).mean()), 4),
            "profit_factor": round(float(wins / losses), 3) if losses > 0 else 99.0,
            "avg": None, "avg_R": round(float(r.mean()), 4),
            "targets": sum(t["outcome"] == "target" for t in trades),
            "stops": sum(t["outcome"] == "stop" for t in trades)}


def _luck(trades: list[dict], sessions_by_symbol: dict, cost: float, reps: int, seed: int = 19) -> dict:
    """Would entering at the same time of day on a random other session of the same
    stock, with the same stop distance and target, have done as well?"""
    rng = np.random.default_rng(seed)
    sims, actual = [], []
    for t in trades:
        sess = sessions_by_symbol.get(t["symbol"], [])
        if len(sess) < 2:
            continue
        k, stop_pct = t["entry_bar"], t["risk"] / t["entry"]
        picks = rng.integers(0, len(sess), size=reps)
        row = []
        for p in picks:
            bars = sess[p][1]
            e = bars[k]["c"]
            risk = e * stop_pct
            s, tgt = e - risk, e + t["target_r"] * risk
            _, r = _play_out(bars, k, e, s, tgt, risk)
            row.append(r - 2 * cost / stop_pct)
        sims.append(row)
        actual.append(net_r(t, cost))
    if not sims:
        return {"p_value": None, "actual_avg_R": None, "random_avg_R": None}
    sims, actual = np.array(sims), np.array(actual)
    rand = sims.mean(axis=0)
    p = float((np.sum(rand >= actual.mean()) + 1) / (reps + 1))
    return {"p_value": round(p, 4), "actual_avg_R": round(float(actual.mean()), 4),
            "random_avg_R": round(float(rand.mean()), 4), "reps": reps}


def day_trade_exam(sessions_by_symbol: dict[str, list[tuple[str, list[dict]]]], original_symbols: Sequence[str],
                   on_check: Callable[[dict], None] | None = None, luck_reps: int = 300) -> dict:
    """The opening-range breakout on every full session, then the stress tests."""
    by_target = {}
    for tr in (1.0, 2.0):
        trades = []
        for sym, sess in sessions_by_symbol.items():
            for d, bars in sess:
                t = orb_trade(d, bars, tr)
                if t:
                    trades.append({**t, "symbol": sym})
        by_target[tr] = trades
    trades = by_target[1.0]
    s = _score(trades, BASE_COST)
    checks = []

    def check(key, name, passed, detail):
        item = {"key": key, "name": name, "passed": bool(passed), "detail": detail}
        checks.append(item)
        if on_check:
            on_check(item)

    days = sorted({t["date"] for t in trades})
    check("basic", "Wins at least half its trades and averages above 0R after 0.05% costs",
          s["trades"] >= 100 and (s["win_rate"] or 0) >= 0.5 and (s["avg_R"] or -1) > 0,
          f"{s['trades']} trades over {len(days)} sessions: won {_p(s['win_rate'])}, average {_r(s['avg_R'])} "
          f"per trade, {s.get('targets', 0)} hit the target and {s.get('stops', 0)} the stop")
    luck = _luck(trades, sessions_by_symbol, BASE_COST, luck_reps)
    check("luck", "Beats entering at the same time of day on random sessions (p under 0.05)",
          luck["p_value"] is not None and luck["p_value"] < 0.05,
          f"breakouts averaged {_r(luck['actual_avg_R'])} vs {_r(luck['random_avg_R'])} for random sessions "
          f"(p = {luck['p_value']})")
    high = _score(trades, HIGH_COST)
    check("costs", "Still averages above 0R at 0.10% costs per side",
          (high["avg_R"] or -1) > 0, f"average {_r(high['avg_R'])} per trade at 0.10% each way")
    two = _score(by_target[2.0], BASE_COST)
    check("target", "Also averages above 0R with a 2R target",
          (two["avg_R"] or -1) > 0, f"2R target: won {_p(two['win_rate'])}, average {_r(two['avg_R'])}")
    per_stock = {}
    for t in trades:
        per_stock.setdefault(t["symbol"], []).append(net_r(t, BASE_COST))
    counted = {k: v for k, v in per_stock.items() if len(v) >= 5}
    good = [k for k, v in counted.items() if np.mean(v) > 0]
    check("breadth", "Averages above 0R in at least 60% of stocks",
          counted and len(good) / len(counted) >= 0.6,
          f"{len(good)} of {len(counted)} stocks (with 5+ trades) averaged above 0R")
    weeks = {}
    for t in trades:
        y, w, _ = date.fromisoformat(t["date"]).isocalendar()
        weeks.setdefault((y, w), []).append(net_r(t, BASE_COST))
    good_w = [k for k, v in weeks.items() if sum(v) > 0]
    check("weeks", "Ends at least 60% of weeks ahead",
          weeks and len(good_w) / len(weeks) >= 0.6, f"ahead in {len(good_w)} of {len(weeks)} weeks")
    new = _score([t for t in trades if t["symbol"] not in set(original_symbols)], BASE_COST)
    check("new_stocks", "Works on the stocks outside Stan's original 8",
          (new["avg_R"] or -1) > 0 and (new["win_rate"] or 0) >= 0.5,
          f"won {_p(new['win_rate'])}, average {_r(new['avg_R'])} on {new['trades']} trades")
    passed = sum(c["passed"] for c in checks)
    return {
        "name": "Opening-range breakout",
        "rule": ("buy when a 5-minute bar closes above the first 15 minutes' high; stop at the range midpoint, "
                 "target 1R, out by 3:55 PM (the Day Trader skill pack's rule)"),
        "unit": "R",
        "cost": BASE_COST,
        "stocks": len(sessions_by_symbol),
        "sessions": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "score": s,
        "two_r": two,
        "luck": luck,
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "verdict": "real_edge" if passed == len(checks) else ("promising" if passed >= len(checks) - 2 else "not_proven"),
    }


# ---------------------------------------------------------------- practice storage
SCHEMA = """
CREATE TABLE IF NOT EXISTS daytrade_practice (
    day TEXT NOT NULL, symbol TEXT NOT NULL, entry_time TEXT, entry REAL, stop REAL, target REAL,
    p_target REAL, p_stop REAL, p_neither REAL, avg_R_sim REAL, verdict TEXT,
    hist_triggered INTEGER, hist_avg_R REAL,
    budget_qty INTEGER, budget_bound_by TEXT, risk_verdict TEXT, risk_reason TEXT, taken INTEGER,
    outcome TEXT, R REAL, net_R REAL, recorded_at TEXT NOT NULL,
    PRIMARY KEY (day, symbol)
);
"""


class PracticeBook:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def has_day(self, day: str) -> bool:
        return self._db.execute("SELECT 1 FROM daytrade_practice WHERE day=? LIMIT 1", (day,)).fetchone() is not None

    def save(self, rows: list[dict]) -> None:
        cols = ["day", "symbol", "entry_time", "entry", "stop", "target", "p_target", "p_stop", "p_neither",
                "avg_R_sim", "verdict", "hist_triggered", "hist_avg_R", "budget_qty", "budget_bound_by",
                "risk_verdict", "risk_reason", "taken", "outcome", "R", "net_R", "recorded_at"]
        for r in rows:
            self._db.execute(
                f"INSERT OR IGNORE INTO daytrade_practice ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [r.get(c) for c in cols])
        self._db.commit()

    def rows(self) -> list[dict]:
        """Newest day first; within a day, in the order the breakouts happened."""
        rows = [dict(r) for r in self._db.execute("SELECT * FROM daytrade_practice")]

        def minutes(r):
            try:
                return timemachine().parse_clock(r["entry_time"], "entry_time")
            except Exception:
                return 0

        rows.sort(key=lambda r: (r["symbol"]))
        rows.sort(key=minutes)
        rows.sort(key=lambda r: r["day"], reverse=True)
        return rows


def practice_stats(rows: list[dict]) -> dict:
    """How Stan's practice trades went, and whether the Time Machine's odds held up."""
    if not rows:
        return {"days": 0, "setups": 0}
    r = np.array([x["net_R"] for x in rows], dtype=float)
    taken = [x for x in rows if x["taken"]]
    leans = [x for x in rows if x["verdict"] and "toward the target" in x["verdict"]]
    flips = [x for x in rows if x["verdict"] and ("coin flip" in x["verdict"] or "break-even" in x["verdict"])]
    against = [x for x in rows if x["verdict"] and "toward the stop" in x["verdict"]]

    def avg(rows_):
        return round(float(np.mean([x["net_R"] for x in rows_])), 3) if rows_ else None

    predicted = float(np.mean([x["p_target"] for x in rows]))
    actual = float(np.mean([x["outcome"] == "target" for x in rows]))
    return {
        "days": len({x["day"] for x in rows}),
        "setups": len(rows),
        "win_rate": round(float((r > 0).mean()), 4),
        "avg_R": round(float(r.mean()), 3),
        "taken": len(taken),
        "taken_avg_R": avg(taken),
        "refused_by_budget": sum(1 for x in rows if not x["budget_qty"]),
        "blocked_by_risk": sum(1 for x in rows if x["budget_qty"] and x["risk_verdict"] != "PASS"),
        "tm_predicted_target": round(predicted, 4),
        "tm_actual_target": round(actual, 4),
        "lean_target_avg_R": avg(leans), "lean_target_n": len(leans),
        "coin_flip_avg_R": avg(flips), "coin_flip_n": len(flips),
        "lean_stop_avg_R": avg(against), "lean_stop_n": len(against),
    }


# ---------------------------------------------------------------- budget and risk, through the pack's own tools
@dataclass(frozen=True)
class PaperRules:
    budget: float = 50.0
    max_loss_per_trade_usd: float = 5.0   # 10% of $50: the spec's per-trade limit
    max_daily_loss_usd: float = 10.0      # 20% of $50: the spec's kill-switch line
    max_trades_per_day: int = 3


def _pack_rules(rules: PaperRules) -> dict:
    cfg = json.loads((SKILL_DIR / "trader-budget" / "config.example.json").read_text())
    cfg.update(budget=rules.budget, max_loss_per_trade_usd=rules.max_loss_per_trade_usd,
               max_order_usd=rules.budget, max_daily_loss_usd=rules.max_daily_loss_usd,
               max_trades_per_day=rules.max_trades_per_day, paper=True, sec_user_agent="")
    return cfg


def budget_and_risk(plan: dict, rules: PaperRules, when: str) -> dict:
    """Run the pack's budget manager and risk manager on one plan, in a scratch folder."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_p, plan_p, orders_p = (os.path.join(tmp, n) for n in ("config.json", "plan.json", "orders.json"))
        Path(cfg_p).write_text(json.dumps(_pack_rules(rules)))
        Path(plan_p).write_text(json.dumps(plan))
        size = subprocess.run(
            [sys.executable, "-I", str(SKILL_DIR / "trader-budget" / "size.py"), "--plan", plan_p,
             "--config", cfg_p, "--now", when, "--json", "--write", orders_p],
            capture_output=True, text=True, timeout=60, cwd=tmp)
        try:
            sizing = json.loads(size.stdout).get("sizing", {})
        except ValueError:
            return {"qty": 0, "bound_by": "error", "risk_verdict": "REFUSED",
                    "risk_reason": (size.stdout or size.stderr).strip().splitlines()[-1:] or ["budget tool failed"]}
        qty = int(sizing.get("qty") or 0)
        if not qty or not os.path.exists(orders_p):
            return {"qty": 0, "bound_by": sizing.get("bound_by"), "risk_verdict": "REFUSED",
                    "risk_reason": f"budget: 0 shares fit ({sizing.get('bound_by') or 'a limit'} decides)"}
        check = subprocess.run(
            [sys.executable, "-I", str(SKILL_DIR / "trader-risk" / "check.py"), "--orders", orders_p,
             "--config", cfg_p, "--now", when, "--today", when[:10], "--json", "--no-write",
             "--state", os.path.join(tmp, "broker.json"), "--portfolio", os.path.join(tmp, "none.json")],
            capture_output=True, text=True, timeout=60, cwd=tmp)
        try:
            res = json.loads(check.stdout)["results"][0]
            verdict, reasons = res["verdict"], res.get("reasons") or []
        except (ValueError, KeyError, IndexError):
            verdict, reasons = "ERROR", [(check.stdout or check.stderr).strip()[:200]]
        return {"qty": qty, "bound_by": sizing.get("bound_by"), "max_loss": sizing.get("max_loss"),
                "risk_verdict": verdict, "risk_reason": "; ".join(reasons)}


# ---------------------------------------------------------------- one evening of practice
def practice_symbol(symbol: str, text: str, day: str, rules: PaperRules, replays: int = 10000) -> dict | None:
    """Walk the pack's chain for one stock's finished session. None when there was no breakout."""
    tm = timemachine()
    end = datetime.combine(date.fromisoformat(day), time(16, 30), tzinfo=ET).astimezone(timezone.utc).replace(tzinfo=None)
    days = sessions_from_text(text, symbol, end)
    sess = days.get(day, {})
    if len(sess) != SESSION_BARS:
        return None  # half day or missing bars
    bars = [sess[i] for i in range(SESSION_BARS)]
    t = orb_trade(day, bars, 1.0)
    if t is None or t["risk"] < MIN_RISK_PCT * t["entry"]:
        return None
    plan = {"created_by": "trader-exit", "symbol": symbol, "side": "buy", "date": day, "setup": "orb",
            "entry": t["entry"], "entry_time": t["entry_time"], "entry_bar": t["entry_bar"], "interval": "5m",
            "opening_range": {"high": t["or_high"], "low": t["or_low"], "minutes": OR_MINUTES},
            "stop": t["stop"], "stop_rule": "or_mid", "target": t["target"], "target_r": 1.0,
            "risk_per_share": round(t["entry"] - t["stop"], 2), "reward_risk": 1.0, "flat_by": UNTIL}
    # The Time Machine only uses sessions before this day, so it can't see the outcome.
    doc = tm.time_machine(text, tm.normalize_plan(plan, "stan"), end, replays=replays, seed=19, until=UNTIL)
    clock = 9 * 60 + 30 + 5 * (t["entry_bar"] + 1)
    br = budget_and_risk(plan, rules, f"{day} {clock // 60:02d}:{clock % 60:02d}")
    return {
        "day": day, "symbol": symbol, "entry_time": t["entry_time"], "entry": t["entry"], "stop": t["stop"],
        "target": t["target"], "p_target": doc["mc"]["p_target"], "p_stop": doc["mc"]["p_stop"],
        "p_neither": doc["mc"]["p_neither"], "avg_R_sim": doc["mc"]["avg_R"], "verdict": doc["verdict"],
        "hist_triggered": doc["history"].get("triggered"), "hist_avg_R": doc["history"].get("avg_R"),
        "budget_qty": br["qty"], "budget_bound_by": br["bound_by"], "risk_verdict": br["risk_verdict"],
        "risk_reason": br["risk_reason"], "taken": 0,
        "outcome": t["outcome"], "R": t["R"], "net_R": round(net_r({**t}, BASE_COST), 3),
        "recorded_at": datetime.now(ET).replace(microsecond=0).isoformat(),
    }


def apply_day_limits(rows: list[dict], rules: PaperRules) -> list[dict]:
    """In trigger order, take passed trades until the day's trade count or loss budget runs out."""
    def minutes(r):
        return timemachine().parse_clock(r["entry_time"], "entry_time")

    used_loss, count = 0.0, 0
    for r in sorted(rows, key=minutes):
        loss = r["budget_qty"] * (r["entry"] - r["stop"]) if r["budget_qty"] else 0.0
        if (r["risk_verdict"] == "PASS" and count < rules.max_trades_per_day
                and used_loss + loss <= rules.max_daily_loss_usd + 1e-9):
            r["taken"] = 1
            count += 1
            used_loss += loss
        elif r["risk_verdict"] == "PASS":
            r["risk_reason"] = "day limit: " + ("3 trades already taken" if count >= rules.max_trades_per_day
                                                else "daily loss budget used up")
    return rows


def practice_day(book: PracticeBook, texts: dict[str, str], day: str, rules: PaperRules = PaperRules(),
                 replays: int = 10000, log: Callable[[str, str], None] | None = None) -> list[dict]:
    if book.has_day(day):
        return []
    rows = []
    for sym, text in texts.items():
        try:
            row = practice_symbol(sym, text, day, rules, replays)
        except Exception as err:  # one bad chart never stops the evening
            if log:
                log("thought", f"Day-trade practice skipped {sym}: {err}")
            continue
        if row:
            rows.append(row)
    rows = apply_day_limits(rows, rules)
    book.save(rows)
    if log:
        if not rows:
            log("thought", f"Day-trade practice for {day}: no opening-range breakouts on my stocks.")
        for r in sorted(rows, key=lambda r: r["entry_time"]):
            status = ("taken" if r["taken"] else
                      "budget said 0 shares" if not r["budget_qty"] else f"not taken ({r['risk_reason'] or r['risk_verdict']})")
            log("win" if r["net_R"] > 0 else "fail",
                f"Day-trade practice {day}: {r['symbol']} broke its opening range at {r['entry_time']} "
                f"(${r['entry']:.2f}, stop ${r['stop']:.2f}, target ${r['target']:.2f}). Time Machine said: "
                f"{r['verdict']} Result: {r['outcome']}, {r['net_R']:+.2f}R after costs. On the $50 plan: {status}.")
    return rows


def _p(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _r(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}R"


# ---------------------------------------------------------------- practice data (made up, for --demo and tests)
def demo_chart_text(symbol: str, sessions: int = 60, seed: int = 7, end: date | None = None) -> str:
    """A Yahoo-shaped 5-minute chart of made-up prices, so the app and tests run offline."""
    import pandas as pd

    import zlib

    rng = np.random.default_rng(zlib.crc32(f"{symbol}:{seed}".encode()))
    end = end or date.today()
    days = pd.bdate_range(end=pd.Timestamp(end), periods=sessions)
    price = float(np.exp(rng.uniform(np.log(8), np.log(400))))  # some cheap enough for a $50 plan
    stamps, o, h, l, c, v = [], [], [], [], [], []
    for d in days:
        drift = 0.0  # no built-in trend: practice data must not hand breakouts a fake edge
        vol = rng.uniform(0.0012, 0.0035)
        for i in range(SESSION_BARS):
            start = datetime.combine(d.date(), time(9, 30), tzinfo=ET) + pd.Timedelta(minutes=5 * i)
            ret = drift + rng.normal(0, vol * (1.8 if i < 6 else 1.0))
            op = price
            cl = price * (1 + ret)
            hi = max(op, cl) * (1 + abs(rng.normal(0, vol / 2)))
            lo = min(op, cl) * (1 - abs(rng.normal(0, vol / 2)))
            stamps.append(int(start.timestamp()))
            o.append(round(op, 2)); h.append(round(hi, 2)); l.append(round(lo, 2)); c.append(round(cl, 2))
            v.append(int(rng.lognormal(10, 0.5)))
            price = cl
        price *= 1 + rng.normal(0, 0.006)  # overnight gap
    last = stamps[-1] + 300
    doc = {"chart": {"result": [{
        "meta": {"symbol": symbol, "dataGranularity": "5m", "regularMarketTime": last},
        "timestamp": stamps,
        "indicators": {"quote": [{"open": o, "high": h, "low": l, "close": c, "volume": v}]},
    }], "error": None}}
    return json.dumps(doc)


def load_charts(symbols: Sequence[str], demo: bool = False, offline: str | None = None,
                log: Callable[[str], None] | None = None) -> dict[str, str]:
    """Chart text per symbol. Symbols that fail are skipped (and logged)."""
    texts = {}
    for sym in symbols:
        try:
            texts[sym] = demo_chart_text(sym) if demo else fetch_chart_text(sym, offline)
        except Exception as err:
            if log:
                log(f"{sym}: {err}")
    return texts


def exam_sessions(texts: dict[str, str], now_utc: datetime | None = None) -> dict[str, list]:
    now_utc = now_utc or datetime.now(timezone.utc).replace(tzinfo=None)
    out = {}
    for sym, text in texts.items():
        try:
            sess = full_sessions(sessions_from_text(text, sym, now_utc))
        except Exception:
            continue
        if sess:
            out[sym] = sess
    return out
