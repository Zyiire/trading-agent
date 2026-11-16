"""QQQ trend bot: Stan's version of the strategy shown in Nick Fischer's "QQQ BOTS" reel.

What the reel shows (and what it doesn't):
  * Three bots on QQQ options (QQQ, QQQ 0G, QQQ TREND), buying calls in up moves and puts in
    down moves, many trades a day, out the same day.
  * Dashboard switches: stop loss, daily loss limit, trend filter, trailing stop, chop filter,
    profit lock, risk level Low/Medium/High, style Scalp/Trend.
  * In his words: it waits, picks the direction, gets in, locks profit on the way with a trailing
    stop and gets out when the move turns.
  * It never shows the entry rule, a backtest, or more than about a week of results, and one of
    the days on screen lost $6,364. So everything below is a reconstruction from those switches,
    not his code, and it has to pass Stan's exam like anything else.

Stan's rule on 5-minute bars (trades the underlying; options are not modelled):
  trend filter  long when close > VWAP, EMA9 > EMA21 and EMA9 rising; short is the mirror
  chop filter   only if the last hour moved in a fairly straight line (efficiency ratio >= 0.30)
  entry         the first bar from 10:00 AM with the trend on and the chop filter clear;
                after an exit, the trend has to reset before the next trade
  stop loss     1.0 / 1.5 / 2.0 ATR for risk Low / Medium / High
  profit lock   once 1R ahead, the stop moves to break-even
  trailing stop then follows the best price by 1.5 / 2.0 / 2.5 ATR
  turn exit     out if a bar closes back across VWAP
  daily limits  at most 3 trades a symbol a day, none after the day reaches -2R; flat by 3:55 PM
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np

from agent.stan.daytrade import LAST_BAR, MIN_RISK_PCT, _p, _r

BASE_COST = 0.0003   # 0.03% per side: QQQ/SPY-class liquidity on 5-minute bars
HIGH_COST = 0.0008
START_BAR = 6        # the 10:00 AM bar: let the open settle ("it waits")
ER_BARS = 12         # one hour of 5-minute bars
ER_MIN = 0.30
ATR_BARS = 12
MAX_TRADES = 3
DAY_STOP_R = -2.0
FOCUS = "QQQ"


@dataclass(frozen=True)
class Risk:
    stop_atr: float
    trail_atr: float


RISK = {"low": Risk(1.0, 1.5), "medium": Risk(1.5, 2.0), "high": Risk(2.0, 2.5)}

RULE = ("trade with the trend on 5-minute bars: long above VWAP with EMA9 over EMA21 and rising, short on the "
        "mirror, only when the last hour moved in a fairly straight line (chop filter); stop 1.5 ATR, stop to "
        "break-even at +1R, then a 2 ATR trailing stop; out on a close back across VWAP or at 3:55 PM; "
        "3 trades and -2R a day at most")


# ---------------------------------------------------------------- indicators for one session
def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a, out = 2 / (n + 1), np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def indicators(bars: list[dict]) -> dict[str, np.ndarray]:
    o = np.array([b["o"] for b in bars], float)
    h = np.array([b["h"] for b in bars], float)
    l = np.array([b["l"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    v = np.array([max(b.get("v") or 0, 1) for b in bars], float)
    tp = (h + l + c) / 3
    vwap = np.cumsum(tp * v) / np.cumsum(v)
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
    atr = np.array([tr[max(0, i - ATR_BARS + 1):i + 1].mean() for i in range(len(tr))])
    er = np.zeros(len(c))
    steps = np.abs(np.diff(c, prepend=c[0]))
    for i in range(ER_BARS, len(c)):
        path = steps[i - ER_BARS + 1:i + 1].sum()
        er[i] = abs(c[i] - c[i - ER_BARS]) / path if path > 0 else 0.0
    return {"o": o, "h": h, "l": l, "c": c, "vwap": vwap, "e9": _ema(c, 9), "e21": _ema(c, 21),
            "atr": atr, "er": er}


def cached(cache: dict, sym: str, day: str, bars: list[dict]) -> dict:
    key = (sym, day)
    if key not in cache:
        cache[key] = indicators(bars)
    return cache[key]


def _trend(ind: dict, i: int) -> int:
    """+1 up, -1 down, 0 neither."""
    c, vw, e9, e21 = ind["c"], ind["vwap"], ind["e9"], ind["e21"]
    if c[i] > vw[i] and e9[i] > e21[i] and e9[i] > e9[i - 1]:
        return 1
    if c[i] < vw[i] and e9[i] < e21[i] and e9[i] < e9[i - 1]:
        return -1
    return 0


def manage(ind: dict, k: int, side: int, entry: float, risk: float, rk: Risk, trail: bool = True
           ) -> tuple[int, float, str]:
    """Bar by bar after the entry: stop, profit lock, trailing stop, turn exit, else 3:55."""
    o, h, l, c, vw, atr = ind["o"], ind["h"], ind["l"], ind["c"], ind["vwap"], ind["atr"]
    stop, best, locked = entry - side * risk, entry, False
    for j in range(k + 1, LAST_BAR + 1):
        hit = l[j] <= stop if side > 0 else h[j] >= stop
        if hit:
            px = min(stop, o[j]) if side > 0 else max(stop, o[j])  # a gap through the stop fills worse
            return j, px, ("lock" if locked and side * (px - entry) >= 0 else "stop")
        best = max(best, h[j]) if side > 0 else min(best, l[j])
        if trail:
            if side * (best - entry) >= risk:
                locked = True
            if locked:
                trail_px = best - side * rk.trail_atr * atr[j]
                stop = max(stop, entry, trail_px) if side > 0 else min(stop, entry, trail_px)
        if side * (c[j] - vw[j]) < 0:
            return j, c[j], "turn"
    return LAST_BAR, c[LAST_BAR], "close"


def session_trades(day: str, bars: list[dict], risk: str = "medium", chop: bool = True, trail: bool = True,
                   ind: dict | None = None) -> list[dict]:
    """Every trade the rule takes in one full session."""
    if len(bars) <= LAST_BAR:
        return []
    ind = ind or indicators(bars)
    rk = RISK[risk]
    out, day_r, i = [], 0.0, max(START_BAR, ER_BARS)
    used = 0  # the trend direction already traded; a new trade needs the trend to reset first
    while i < LAST_BAR and len(out) < MAX_TRADES and day_r > DAY_STOP_R:
        side = _trend(ind, i)
        if side != used:
            used = 0
        if side != 0 and side != used and (not chop or ind["er"][i] >= ER_MIN):
            entry = float(ind["c"][i])
            r = rk.stop_atr * float(ind["atr"][i])
            if r >= entry * MIN_RISK_PCT:
                j, px, why = manage(ind, i, side, entry, r, rk, trail)
                R = side * (px - entry) / r
                out.append({"date": day, "side": "long" if side > 0 else "short", "entry_bar": i,
                            "exit_bar": j, "entry": round(entry, 4), "exit": round(float(px), 4),
                            "risk": round(r, 4), "outcome": why, "R": round(float(R), 3)})
                day_r += R
                used = _trend(ind, j) if j < len(ind["c"]) else 0
                used = used if used == side else 0
                i = j + 1
                continue
        i += 1
    return out


# ---------------------------------------------------------------- scoring
def net_r(t: dict, cost: float) -> float:
    return t["R"] - 2 * cost * t["entry"] / t["risk"]


def score(trades: list[dict], cost: float = BASE_COST) -> dict:
    r = np.array([net_r(t, cost) for t in trades])
    if not len(r):
        return {"trades": 0, "win_rate": None, "profit_factor": None, "avg": None, "avg_R": None}
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": int(len(r)), "win_rate": round(float((r > 0).mean()), 4),
            "profit_factor": round(float(wins / losses), 3) if losses > 0 else 99.0,
            "avg": None, "avg_R": round(float(r.mean()), 4), "total_R": round(float(r.sum()), 2),
            "longs": sum(t["side"] == "long" for t in trades), "shorts": sum(t["side"] == "short" for t in trades)}


def max_drawdown_r(trades: list[dict], cost: float) -> tuple[float, float]:
    """(worst peak-to-trough in R, total R) for trades in time order."""
    eq = np.cumsum([net_r(t, cost) for t in sorted(trades, key=lambda t: (t["date"], t["entry_bar"]))])
    if not len(eq):
        return 0.0, 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float((peak - eq).max()), float(eq[-1])


def all_trades(sessions_by_symbol: dict, cache: dict, **kw) -> list[dict]:
    out = []
    for sym, sess in sessions_by_symbol.items():
        for d, bars in sess:
            ind = cached(cache, sym, d, bars)
            out += [{**t, "symbol": sym} for t in session_trades(d, bars, ind=ind, **kw)]
    return out


def _luck(trades: list[dict], sessions_by_symbol: dict, cache: dict, cost: float, reps: int,
          seed: int = 23, sample: int = 600) -> dict:
    """Same symbol, same time of day, same direction, same stop and exits, on a random other session.
    If that does as well, the trend and chop filters aren't adding anything."""
    rng = np.random.default_rng(seed)
    rk = RISK["medium"]
    act = float(np.mean([net_r(t, cost) for t in trades])) if trades else None  # every trade, not the sample
    if len(trades) > sample:  # a fixed random sample keeps 40+ symbols to a minute or two
        trades = [trades[i] for i in sorted(rng.choice(len(trades), size=sample, replace=False))]
    sims = []
    for t in trades:
        sess = sessions_by_symbol.get(t["symbol"], [])
        if len(sess) < 2:
            continue
        k, side, stop_pct = t["entry_bar"], (1 if t["side"] == "long" else -1), t["risk"] / t["entry"]
        row = []
        for p in rng.integers(0, len(sess), size=reps):
            d, bars = sess[p]
            ind = cached(cache, t["symbol"], d, bars)
            e = float(ind["c"][k])
            r = e * stop_pct
            _, px, _ = manage(ind, k, side, e, r, rk)
            row.append(side * (px - e) / r - 2 * cost / stop_pct)
        sims.append(row)
    if not sims:
        return {"p_value": None, "actual_avg_R": None, "random_avg_R": None}
    rand = np.array(sims).mean(axis=0)
    p = float((np.sum(rand >= act) + 1) / (reps + 1))
    return {"p_value": round(p, 4), "actual_avg_R": round(act, 4), "random_avg_R": round(float(rand.mean()), 4),
            "reps": reps}


# ---------------------------------------------------------------- the exam
def trend_exam(sessions_by_symbol: dict[str, list[tuple[str, list[dict]]]],
               on_check: Callable[[dict], None] | None = None, luck_reps: int = 200) -> dict:
    cache: dict = {}
    trades = all_trades(sessions_by_symbol, cache)
    s = score(trades)
    checks = []

    def check(key, name, passed, detail):
        item = {"key": key, "name": name, "passed": bool(passed), "detail": detail}
        checks.append(item)
        if on_check:
            on_check(item)

    days = sorted({t["date"] for t in trades} | {d for v in sessions_by_symbol.values() for d, _ in v})
    check("basic", "Makes more than it loses after 0.03% costs (profit factor 1.2+, 60+ trades)",
          s["trades"] >= 60 and (s["avg_R"] or -1) > 0 and (s["profit_factor"] or 0) >= 1.2,
          f"{s['trades']} trades ({s.get('longs', 0)} long, {s.get('shorts', 0)} short): won {_p(s['win_rate'])}, "
          f"average {_r(s['avg_R'])}, profit factor {s['profit_factor']}")

    luck = _luck(trades, sessions_by_symbol, cache, BASE_COST, luck_reps)
    check("luck", "Profitable, and beats the same trades taken at random on other days (p under 0.05)",
          luck["p_value"] is not None and luck["p_value"] < 0.05 and (luck["actual_avg_R"] or -1) > 0,
          f"its trades averaged {_r(luck['actual_avg_R'])} vs {_r(luck['random_avg_R'])} at random (p = {luck['p_value']})")

    high = score(trades, HIGH_COST)
    check("costs", "Still ahead at 0.08% costs per side",
          (high["avg_R"] or -1) > 0, f"average {_r(high['avg_R'])} at 0.08% each way")

    longs = score([t for t in trades if t["side"] == "long"])
    shorts = score([t for t in trades if t["side"] == "short"])
    check("both_sides", "Makes money on the short side too, not just guessing up",
          longs["trades"] >= 20 and shorts["trades"] >= 20 and (longs["avg_R"] or -1) > 0 and (shorts["avg_R"] or -1) > 0,
          f"longs {_r(longs['avg_R'])} on {longs['trades']}, shorts {_r(shorts['avg_R'])} on {shorts['trades']}")

    levels = {name: score(all_trades(sessions_by_symbol, cache, risk=name)) for name in ("low", "high")}
    levels["medium"] = s
    check("risk_levels", "Works at Low, Medium and High risk, not one lucky setting",
          all((v["avg_R"] or -1) > 0 for v in levels.values()),
          ", ".join(f"{k} {_r(levels[k]['avg_R'])}" for k in ("low", "medium", "high")))

    raw = score(all_trades(sessions_by_symbol, cache, chop=False))
    check("chop_filter", "The chop filter earns its keep (better with it than without)",
          (s["avg_R"] or -9) > (raw["avg_R"] or -9),
          f"with it {_r(s['avg_R'])} on {s['trades']} trades, without it {_r(raw['avg_R'])} on {raw['trades']}")

    weeks: dict = {}
    for t in trades:
        y, w, _ = date.fromisoformat(t["date"]).isocalendar()
        weeks.setdefault((y, w), []).append(net_r(t, BASE_COST))
    good_w = [k for k, v in weeks.items() if sum(v) > 0]
    check("weeks", "Ends at least 55% of weeks ahead",
          weeks and len(good_w) / len(weeks) >= 0.55, f"ahead in {len(good_w)} of {len(weeks)} weeks")

    per_sym: dict = {}
    for t in trades:
        per_sym.setdefault(t["symbol"], []).append(net_r(t, BASE_COST))
    counted = {k: v for k, v in per_sym.items() if len(v) >= 5 and k != FOCUS}
    good = [k for k, v in counted.items() if np.mean(v) > 0]
    focus = score([t for t in trades if t["symbol"] == FOCUS])
    check("breadth", "Works on QQQ and on at least 60% of the other symbols",
          (focus["avg_R"] or -1) > 0 and counted and len(good) / len(counted) >= 0.6,
          f"QQQ {_r(focus['avg_R'])} on {focus['trades']} trades; {len(good)} of {len(counted)} others ahead")

    qqq = [t for t in trades if t["symbol"] == FOCUS] or trades
    dd, total = max_drawdown_r(qqq, BASE_COST)
    check("drawdown", "Its worst losing stretch is smaller than its total gain",
          total > 0 and dd < total, f"{FOCUS if qqq is not trades else 'all symbols'}: worst drop {dd:.1f}R, "
                                    f"total {total:+.1f}R")

    passed = sum(c["passed"] for c in checks)
    return {
        "name": "QQQ trend bot",
        "rule": RULE,
        "source": "rebuilt from the controls in Nick Fischer's QQQ BOTS reel; his exact rules aren't public",
        "unit": "R",
        "cost": BASE_COST,
        "stocks": len(sessions_by_symbol),
        "sessions": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "score": s,
        "focus": focus,
        "luck": luck,
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "verdict": "real_edge" if passed == len(checks) else ("promising" if passed >= len(checks) - 2 else "not_proven"),
    }
