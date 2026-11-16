"""Putting the four parts together: context per symbol, the signal at each closed bar, and trades.

`signal_at(ctx, i)` is the only place a decision is made, and it reads nothing after bar i:
  * trend[i] comes from pivots with confirmed_at <= i (structure.trend_by_bar)
  * levels come from earlier sessions, premarket, and pivots confirmed by i (levels.levels_at)
  * ATR at i and the volume average before i use bars <= i
  * the pattern looks at bars i-1 and i, both closed
tests/test_price_action.py checks this by re-running on data cut off at bar i.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.stan.price_action import levels as lv
from agent.stan.price_action import patterns as pt
from agent.stan.price_action import risk as rk
from agent.stan.price_action import structure as st
from agent.stan.price_action.config import PriceActionConfig
from agent.stan.price_action.series import Series, atr, build_htf, from_sessions, prior_mean

SWING_LOOKBACK = 6  # LTF bars for the optional swing stop (30 minutes)


@dataclass
class Context:
    s: Series
    cfg: PriceActionConfig
    pivots: list[st.Pivot]
    trend: np.ndarray
    atr: np.ndarray
    avg_vol: np.ndarray
    day_levels: dict[int, list[lv.Level]]


@dataclass(frozen=True)
class Signal:
    bar: int
    side: int
    pattern: str
    level: lv.Level
    structure: float     # the wick low/high the stop goes beyond
    swing: float
    levels: tuple[lv.Level, ...]
    buffer: float


def context(s: Series, cfg: PriceActionConfig) -> Context:
    htf = build_htf(s, cfg.htf_bars)
    pivots = st.find_pivots(htf, cfg.pivot_n)
    return Context(s, cfg, pivots, st.trend_by_bar(pivots, len(s), cfg.structure_pivots),
                   atr(s, cfg.atr_period), prior_mean(s.v, cfg.vol_period), lv.day_levels(s, cfg))


def _bar(s: Series, i: int) -> pt.Bar:
    return pt.Bar(float(s.o[i]), float(s.h[i]), float(s.l[i]), float(s.c[i]), float(s.v[i]))


def signal_at(ctx: Context, i: int) -> Signal | None:
    """A trade signal on the close of bar i, or None. Reads only bars 0..i."""
    s, cfg = ctx.s, ctx.cfg
    if i < 1 or s.day_idx[i - 1] != s.day_idx[i]:
        return None  # the previous candle must be from the same session
    if not (cfg.first_entry_slot <= s.slot[i] < cfg.flatten_slot):
        return None
    side = int(ctx.trend[i])
    if side == st.CHOP:
        return None
    levels = lv.levels_at(i, ctx.day_levels[int(s.day_idx[i])], ctx.pivots, cfg)
    if not levels:
        return None
    buf = lv.zone_buffer(float(s.c[i]), float(ctx.atr[i]), cfg)
    prev, b = _bar(s, i - 1), _bar(s, i)
    # support below the close for longs, resistance above for shorts, nearest first
    tested = [x for x in levels if (side > 0 and min(b.l, prev.l) <= x.price + buf and b.c > x.price)
              or (side < 0 and max(b.h, prev.h) >= x.price - buf and b.c < x.price)]
    for level in sorted(tested, key=lambda x: abs(b.c - x.price)):
        hit = pt.confirm(prev, b, side, level.price, buf, float(ctx.avg_vol[i]), cfg)
        if hit:
            swing = st.ltf_swing(s.l if side > 0 else s.h, i, SWING_LOOKBACK, lowest=side > 0)
            return Signal(i, side, hit[0], level, hit[1], swing, tuple(levels), buf)
    return None


def trades_for(ctx: Context) -> list[dict]:
    """Walk every session: one position at a time, at most max_trades_per_day, flat by the flatten bar."""
    s, cfg = ctx.s, ctx.cfg
    out, i, n = [], 1, len(s)
    count: dict[int, int] = {}
    while i < n:
        d = int(s.day_idx[i])
        sig = signal_at(ctx, i) if count.get(d, 0) < cfg.max_trades_per_day else None
        if sig is None:
            i += 1
            continue
        last = i + (cfg.flatten_slot - int(s.slot[i]))     # this session's flatten bar
        if cfg.fill == "next_open":
            k, entry = i + 1, float(s.o[i + 1])
        else:
            k, entry = i + 1, float(s.c[i])
        if k > last:
            i += 1
            continue
        opposing = lv.next_opposing(list(sig.levels), entry, sig.side, sig.buffer)
        p = rk.plan(sig.side, entry, sig.structure, sig.swing, opposing, cfg)
        if p is None:
            i += 1
            continue
        j, px, why = rk.walk(p, s.o, s.h, s.l, s.c, k, last, cfg)
        out.append({"symbol": s.symbol, "date": s.days[d], "side": "long" if p.side > 0 else "short",
                    "pattern": sig.pattern, "level": sig.level.name, "level_price": round(sig.level.price, 4),
                    "signal_slot": int(s.slot[i]), "entry_bar": int(s.slot[k]), "exit_bar": int(s.slot[j]),
                    "entry": p.entry, "stop": p.stop, "target": p.target, "target_kind": p.target_kind,
                    "risk": p.risk, "shares": p.shares, "exit": round(float(px), 4), "outcome": why,
                    "R": round(p.side * (float(px) - p.entry) / p.risk, 3)})
        count[d] = count.get(d, 0) + 1
        i = j + 1
    return out


def run(sessions_by_symbol: dict[str, list[tuple[str, list[dict]]]], cfg: PriceActionConfig = PriceActionConfig(),
        cache: dict | None = None) -> list[dict]:
    """Trades on every symbol. `cache` (symbol -> Series) avoids rebuilding the arrays."""
    trades = []
    for sym, sess in sessions_by_symbol.items():
        s = cache.setdefault(sym, None) if cache is not None else None
        if s is None:
            s = from_sessions(sym, sess)
            if cache is not None:
                cache[sym] = s
        if len(s) > 1:
            trades += trades_for(context(s, cfg))
    return trades
