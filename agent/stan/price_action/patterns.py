"""3. Confirmation candles at a level. Each check looks only at the candle that just closed and the
one before it. `side` is +1 for a long at support, -1 for a short at resistance.

Every function returns the structure extreme the stop goes beyond (a low for longs, a high for
shorts), or None when the pattern is not there.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.stan.price_action.config import PriceActionConfig


@dataclass(frozen=True)
class Bar:
    o: float
    h: float
    l: float
    c: float
    v: float


def pin_bar(b: Bar, side: int, level: float, cfg: PriceActionConfig) -> float | None:
    """Rejection: the wick pierces the level, the close is back on the trend side, the rejecting
    wick is at least `pin_wick_body` times the body, and the close sits in the top (bottom) 30%."""
    rng = b.h - b.l
    if rng <= 0:
        return None
    body = max(abs(b.c - b.o), rng * 1e-3)  # a doji counts, with a tiny body
    if side > 0:
        wick = min(b.o, b.c) - b.l
        ok = b.l <= level < b.c and wick >= cfg.pin_wick_body * body and (b.c - b.l) / rng >= 1 - cfg.pin_close_frac
        return b.l if ok else None
    wick = b.h - max(b.o, b.c)
    ok = b.h >= level > b.c and wick >= cfg.pin_wick_body * body and (b.h - b.c) / rng >= 1 - cfg.pin_close_frac
    return b.h if ok else None


def engulfing(prev: Bar, b: Bar, side: int, level: float, buffer: float, avg_vol: float,
              cfg: PriceActionConfig) -> float | None:
    """The candle's body swallows the prior body in the trend direction, it touched the level's zone,
    and its volume beats `engulf_vol_mult` times the average of the 20 bars before it."""
    if not avg_vol or avg_vol != avg_vol or b.v <= cfg.engulf_vol_mult * avg_vol:  # nan-safe
        return None
    lo_p, hi_p = min(prev.o, prev.c), max(prev.o, prev.c)
    if side > 0:
        ok = b.c > b.o and b.o <= lo_p and b.c >= hi_p and (b.c - b.o) > (hi_p - lo_p) \
            and min(b.l, prev.l) <= level + buffer and b.c > level
        return min(b.l, prev.l) if ok else None
    ok = b.c < b.o and b.o >= hi_p and b.c <= lo_p and (b.o - b.c) > (hi_p - lo_p) \
        and max(b.h, prev.h) >= level - buffer and b.c < level
    return max(b.h, prev.h) if ok else None


def failed_breakout(prev: Bar, b: Bar, side: int, level: float) -> float | None:
    """Liquidity sweep: the previous candle closed beyond the level, this one closes back inside."""
    if side > 0:
        return min(prev.l, b.l) if prev.c < level < b.c else None
    return max(prev.h, b.h) if prev.c > level > b.c else None


def confirm(prev: Bar, b: Bar, side: int, level: float, buffer: float, avg_vol: float,
            cfg: PriceActionConfig) -> tuple[str, float] | None:
    """The first configured pattern present at this level: (name, structure extreme)."""
    for name in cfg.patterns:
        if name == "pin":
            x = pin_bar(b, side, level, cfg)
        elif name == "engulf":
            x = engulfing(prev, b, side, level, buffer, avg_vol, cfg)
        elif name == "sweep":
            x = failed_breakout(prev, b, side, level)
        else:
            raise ValueError(f"unknown pattern {name}")
        if x is not None:
            return name, x
    return None
