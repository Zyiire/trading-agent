"""Level play: Stan's rebuild of TZ (@tz.fund)'s "FVG plus LVN nodes" strategy, with its exam.

What his reels show:
  * "Here's a simple trading strategy, 3 step confluence": he checks options volume (puts vs calls by
    expiry), marks levels with TradingView's anchored / fixed-range volume profile, and looks for a
    fair value gap (FVG) at those levels.
  * "The best trading strategy: FVG plus LVN nodes" (low-volume nodes on the volume profile).
  * His QQQ call on Nov 12: "If buyers lose 680, likely put side. If buyers can reclaim 680, likely
    a bullish recovery. Next target above 682.20, 684." He bought same-day QQQ puts and sold them all
    at the first target.
  The reels never spell out exact settings, and the options-flow step can't be tested (there's no free
    history of options volume), so these rules are a reconstruction, not his code.

Stan's rule, on 5-minute bars:
  levels   a volume profile of the previous 3 sessions: the busiest price (POC), the edges of the
           70% value area, and the low-volume nodes (LVNs): prices the market passed through quickly
  trigger  "lose the level": a bar closes below a level after the previous bar closed at or above it,
           with a bearish FVG in the last 3 bars (a candle whose high is below the low two candles
           earlier) -> short. "Reclaim the level" is the mirror -> long.
  stop     2 cents beyond the extreme of the last 3 bars
  target   the next profile level in the trade's direction (his "next target"), at least 1R away and
           at most 3R; stop to break-even at 1R
  session  entries from 9:45 AM, flat by 3:45 PM, one position at a time, at most 3 trades a day
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Callable

import numpy as np

from agent.stan.daytrade import _p, _r
from agent.stan.price_action import risk as rk
from agent.stan.price_action.config import PriceActionConfig
from agent.stan.price_action.exam import BASE_COST, HIGH_COST, _luck, net_r, score
from agent.stan.price_action.series import Series, from_sessions

FOCUS = "QQQ"


@dataclass(frozen=True)
class LevelConfig:
    profile_sessions: int = 3       # sessions in the volume profile
    bin_frac: float = 0.0005        # profile bin width as a fraction of price (about $0.30 on QQQ)
    value_area: float = 0.70
    lvn_depth: float = 0.5          # an LVN stretch is below half the profile's average volume
    need_fvg: bool = True
    fvg_lookback: int = 3           # an FVG among the last 3 bars counts
    trade: PriceActionConfig = PriceActionConfig(rr_target=3.0, min_room_r=1.0, stop_buffer=0.02,
                                                 breakeven_at_r=1.0, max_trades_per_day=3)


RULE = ("volume-profile levels from the last 3 sessions (busiest price, value-area edges, low-volume nodes); "
        "short when a 5-minute bar closes below a level with a bearish fair value gap, long when it closes back "
        "above with a bullish one; stop past the last 3 bars, target the next level (1R to 3R), break-even at "
        "1R, entries from 9:45 AM, flat by 3:45 PM")


# ---------------------------------------------------------------- volume profile
def profile_levels(h: np.ndarray, l: np.ndarray, v: np.ndarray, cfg: LevelConfig) -> list[tuple[str, float]]:
    """Levels from a volume profile of the given bars. Each bar's volume is spread evenly over its range."""
    if len(h) < 10:
        return []
    lo, hi = float(l.min()), float(h.max())
    width = max(lo * cfg.bin_frac, 0.01)
    n = int(np.ceil((hi - lo) / width)) + 1
    vol = np.zeros(n)
    for hh, ll, vv in zip(h, l, v):
        a, b = int((ll - lo) / width), int((hh - lo) / width)
        vol[a:b + 1] += vv / (b - a + 1)
    price = lo + (np.arange(n) + 0.5) * width
    poc = int(vol.argmax())
    out = [("POC", float(price[poc]))]
    # value area: grow out from the POC until it holds 70% of the volume
    a = b = poc
    total, inside = vol.sum(), vol[poc]
    while inside < cfg.value_area * total and (a > 0 or b < n - 1):
        up = vol[b + 1] if b < n - 1 else -1
        dn = vol[a - 1] if a > 0 else -1
        if up >= dn:
            b += 1
            inside += vol[b]
        else:
            a -= 1
            inside += vol[a]
    out += [("VAH", float(price[b])), ("VAL", float(price[a]))]
    # low-volume nodes: the middle of each quiet stretch (below half the average) with busier
    # prices on both sides, where the market passed through quickly
    smooth = np.convolve(vol, np.ones(3) / 3, mode="same")
    quiet = smooth < cfg.lvn_depth * vol.mean()
    k = 0
    while k < n:
        if quiet[k]:
            j = k
            while j + 1 < n and quiet[j + 1]:
                j += 1
            if k > 0 and j < n - 1 and j - k + 1 >= 2:    # an interior valley at least two bins wide
                out.append(("LVN", float(price[(k + j) // 2])))
            k = j + 1
        else:
            k += 1
    return out


def session_levels(s: Series, cfg: LevelConfig) -> dict[int, list[tuple[str, float]]]:
    """Levels for each session, built only from the sessions before it."""
    out = {}
    for k in range(len(s.days)):
        if k < cfg.profile_sessions:
            out[k] = []
            continue
        m = (s.day_idx >= k - cfg.profile_sessions) & (s.day_idx < k)
        out[k] = profile_levels(s.h[m], s.l[m], s.v[m], cfg)
    return out


def fvg(h: np.ndarray, l: np.ndarray, i: int, side: int, lookback: int) -> bool:
    """A fair value gap ending at or before bar i (within `lookback` bars): for a short, a bar whose
    high is below the low two bars earlier; for a long, a bar whose low is above the high two bars earlier."""
    for j in range(i, max(i - lookback, 1), -1):
        if side < 0 and h[j] < l[j - 2]:
            return True
        if side > 0 and l[j] > h[j - 2]:
            return True
    return False


# ---------------------------------------------------------------- signals and trades
@dataclass(frozen=True)
class Signal:
    bar: int
    side: int
    level_name: str
    level: float
    structure: float
    target: float | None


def signal_at(s: Series, i: int, levels: list[tuple[str, float]], cfg: LevelConfig) -> Signal | None:
    """A signal on the close of bar i, from bars 0..i and the session's levels only."""
    t = cfg.trade
    if i < 3 or s.day_idx[i - 1] != s.day_idx[i] or s.day_idx[i - 2] != s.day_idx[i]:
        return None
    if not (t.first_entry_slot <= s.slot[i] < t.flatten_slot) or not levels:
        return None
    c0, c1 = s.c[i - 1], s.c[i]
    for name, lv in sorted(levels, key=lambda x: abs(c1 - x[1])):
        side = -1 if c0 >= lv > c1 else (1 if c0 <= lv < c1 else 0)
        if not side:
            continue
        if cfg.need_fvg and not fvg(s.h, s.l, i, side, cfg.fvg_lookback):
            continue
        lo3, hi3 = float(s.l[i - 2:i + 1].min()), float(s.h[i - 2:i + 1].max())
        beyond = [p for _, p in levels if side * (p - c1) > 0]
        target = (min(beyond) if side > 0 else max(beyond)) if beyond else None
        return Signal(i, side, name, lv, lo3 if side > 0 else hi3, target)
    return None


def trades_for(s: Series, cfg: LevelConfig = LevelConfig()) -> list[dict]:
    t = cfg.trade
    lv = session_levels(s, cfg)
    out, i, n, count = [], 3, len(s), {}
    while i < n - 1:
        d = int(s.day_idx[i])
        sig = signal_at(s, i, lv[d], cfg) if count.get(d, 0) < t.max_trades_per_day else None
        if sig is None:
            i += 1
            continue
        last = i + (t.flatten_slot - int(s.slot[i]))
        k = i + 1
        if k > last or s.day_idx[k] != d:
            i += 1
            continue
        entry = float(s.o[k])
        p = rk.plan(sig.side, entry, sig.structure, None, sig.target, t)
        if p is None:
            i += 1
            continue
        j, px, why = rk.walk(p, s.o, s.h, s.l, s.c, k, last, t)
        out.append({"symbol": s.symbol, "date": s.days[d], "side": "long" if p.side > 0 else "short",
                    "pattern": "reclaim" if p.side > 0 else "lose", "level": sig.level_name,
                    "level_price": round(sig.level, 4), "signal_slot": int(s.slot[i]), "entry_bar": int(s.slot[k]),
                    "exit_bar": int(s.slot[j]), "entry": p.entry, "stop": p.stop, "target": p.target,
                    "target_kind": p.target_kind, "risk": p.risk, "shares": p.shares, "exit": round(float(px), 4),
                    "outcome": why, "R": round(p.side * (float(px) - p.entry) / p.risk, 3)})
        count[d] = count.get(d, 0) + 1
        i = j + 1
    return out


def run(sessions_by_symbol: dict, cfg: LevelConfig = LevelConfig(), cache: dict | None = None) -> list[dict]:
    trades = []
    for sym, sess in sessions_by_symbol.items():
        s = cache.get(sym) if cache is not None else None
        if s is None:
            s = from_sessions(sym, sess)
            if cache is not None:
                cache[sym] = s
        if len(s) > 10:
            trades += trades_for(s, cfg)
    return trades


# ---------------------------------------------------------------- the exam
def levels_exam(sessions_by_symbol: dict, cfg: LevelConfig = LevelConfig(),
                on_check: Callable[[dict], None] | None = None, luck_reps: int = 200) -> dict:
    cache: dict = {}
    trades = run(sessions_by_symbol, cfg, cache)
    s = score(trades)
    checks = []

    def check(key, name, passed, detail):
        item = {"key": key, "name": name, "passed": bool(passed), "detail": detail}
        checks.append(item)
        if on_check:
            on_check(item)

    days = sorted({d for v in sessions_by_symbol.values() for d, _ in v})
    check("basic", "Makes more than it loses after 0.05% costs (profit factor 1.2+, 60+ trades)",
          s["trades"] >= 60 and (s["avg_R"] or -1) > 0 and (s["profit_factor"] or 0) >= 1.2,
          f"{s['trades']} trades ({s.get('longs', 0)} long, {s.get('shorts', 0)} short): won {_p(s['win_rate'])}, "
          f"average {_r(s['avg_R'])}, profit factor {s['profit_factor']}")
    luck = _luck(trades, cache, cfg.trade, luck_reps)
    check("luck", "Profitable, and beats the same trades placed on random days (p under 0.05)",
          luck["p_value"] is not None and luck["p_value"] < 0.05 and (luck["actual_avg_R"] or -1) > 0,
          f"its trades averaged {_r(luck['actual_avg_R'])} vs {_r(luck['random_avg_R'])} at random (p = {luck['p_value']})")
    high = score(trades, HIGH_COST)
    check("costs", "Still ahead at 0.10% costs per side", (high["avg_R"] or -1) > 0,
          f"average {_r(high['avg_R'])} at 0.10% each way")
    longs, shorts = (score([t for t in trades if t["side"] == x]) for x in ("long", "short"))
    check("both_sides", "Makes money losing levels (puts) and reclaiming them (calls)",
          longs["trades"] >= 20 and shorts["trades"] >= 20 and (longs["avg_R"] or -1) > 0 and (shorts["avg_R"] or -1) > 0,
          f"reclaims {_r(longs['avg_R'])} on {longs['trades']}, losses of a level {_r(shorts['avg_R'])} on {shorts['trades']}")
    raw = score(run(sessions_by_symbol, replace(cfg, need_fvg=False), cache))
    check("fvg", "The fair value gap earns its keep (better with it than without)",
          (s["avg_R"] or -9) > (raw["avg_R"] or -9),
          f"with the FVG {_r(s['avg_R'])} on {s['trades']}, without {_r(raw['avg_R'])} on {raw['trades']}")
    alt = {n: score(run(sessions_by_symbol, replace(cfg, profile_sessions=n), cache)) for n in (1, 5)}
    check("profiles", "Also ahead with a 1-session and a 5-session profile, not one lucky setting",
          all((v["avg_R"] or -1) > 0 for v in alt.values()),
          f"1 session {_r(alt[1]['avg_R'])}, 3 sessions {_r(s['avg_R'])}, 5 sessions {_r(alt[5]['avg_R'])}")
    weeks: dict = {}
    for t in trades:
        y, w, _ = date.fromisoformat(t["date"]).isocalendar()
        weeks.setdefault((y, w), []).append(net_r(t, BASE_COST))
    good_w = [k for k, v in weeks.items() if sum(v) > 0]
    check("weeks", "Ends at least 55% of weeks ahead", weeks and len(good_w) / len(weeks) >= 0.55,
          f"ahead in {len(good_w)} of {len(weeks)} weeks")
    per: dict = {}
    for t in trades:
        per.setdefault(t["symbol"], []).append(net_r(t, BASE_COST))
    others = {k: v for k, v in per.items() if len(v) >= 5 and k != FOCUS}
    good = [k for k, v in others.items() if np.mean(v) > 0]
    focus = score([t for t in trades if t["symbol"] == FOCUS])
    check("breadth", "Works on QQQ and on at least 60% of other stocks",
          (focus["avg_R"] or -1) > 0 and others and len(good) / len(others) >= 0.6,
          f"QQQ {_r(focus['avg_R'])} on {focus['trades']} trades; {len(good)} of {len(others)} others ahead")
    eq = np.cumsum([net_r(t, BASE_COST) for t in sorted(trades, key=lambda t: (t["date"], t["entry_bar"]))])
    dd = float((np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq).max()) if len(eq) else 0.0
    total = float(eq[-1]) if len(eq) else 0.0
    check("drawdown", "Its worst losing stretch is smaller than its total gain", total > 0 and dd < total,
          f"worst drop {dd:.1f}R, total {total:+.1f}R")
    passed = sum(c["passed"] for c in checks)
    return {"name": "Level play (FVG + volume profile)", "rule": RULE, "unit": "R", "cost": BASE_COST,
            "source": "rebuilt from TZ (@tz.fund)'s reels; his exact settings and options-flow step aren't public",
            "stocks": len(sessions_by_symbol), "sessions": len(days),
            "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
            "score": s, "focus": focus, "luck": luck, "checks": checks, "checks_passed": passed,
            "checks_total": len(checks),
            "verdict": "real_edge" if passed == len(checks) else ("promising" if passed >= len(checks) - 2 else "not_proven")}
