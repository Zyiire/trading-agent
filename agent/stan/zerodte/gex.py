"""Gamma exposure (GEX): where option dealers' hedging tends to pin or push the price.

The usual convention, used here: dealers are long the calls the public sold them and short the puts
the public bought, so call gamma counts positive and put gamma negative. Per strike:

    exposure = sign x gamma x open interest x 100 x spot^2 x 1%     (dollars of hedging per 1% move)

  call wall    the strike with the most call exposure: often a ceiling
  put wall     the strike with the most put exposure: often a floor
  zero gamma   the price where total exposure flips sign. Above it dealers damp moves (calmer,
               mean-reverting days); below it they chase moves (faster, trending days).

This is a model, not a fact: real dealer positioning isn't public, and the sign convention is
wrong for some strikes. Phase 3 tests whether these levels matter for QQQ at all.
Gamma is recomputed from each contract's IV, so every number here comes from one formula.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time

from agent.stan.zerodte.bars import ET

MIN_YEARS = 1 / (365 * 24 * 60)  # one minute


def years_to_expiry(now_ts: int, expiry: date) -> float | None:
    """Time left until the 4:00 PM close on `expiry`, in years; None once it has expired."""
    end = datetime.combine(expiry, time(16, 0), tzinfo=ET).timestamp()
    left = end - now_ts
    return None if left <= 0 else max(left / (365 * 24 * 3600), MIN_YEARS)


def bs_gamma(spot: float, strike: float, years: float, iv: float, rate: float = 0.0) -> float:
    if spot <= 0 or strike <= 0 or years <= 0 or iv <= 0:
        return 0.0
    st = iv * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2) * years) / st
    return math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) / (spot * st)


def _iv(r, fallback: float | None) -> float | None:
    return r.iv if r.iv and 0.01 < r.iv < 5 else fallback


def by_strike(rows, spot: float, now_ts: int, iv_fallback: float | None = None) -> dict[float, dict]:
    out: dict[float, dict] = {}
    for r in rows:
        years, iv = years_to_expiry(now_ts, r.expiry), _iv(r, iv_fallback)
        if not r.oi or years is None or iv is None:
            continue
        g = bs_gamma(spot, r.strike, years, iv) * r.oi * 100 * spot * spot * 0.01
        cell = out.setdefault(r.strike, {"call": 0.0, "put": 0.0})
        cell[r.type] += g if r.type == "call" else -g
    for cell in out.values():
        cell["net"] = cell["call"] + cell["put"]
    return dict(sorted(out.items()))


def total(rows, spot: float, now_ts: int, iv_fallback: float | None = None) -> float:
    return sum(c["net"] for c in by_strike(rows, spot, now_ts, iv_fallback).values())


def zero_gamma(rows, spot: float, now_ts: int, iv_fallback: float | None = None,
               span: float = 0.05, steps: int = 100) -> float | None:
    """The price nearest `spot` (within +/- span) where total exposure changes sign, or None."""
    grid = [spot * (1 - span + 2 * span * i / steps) for i in range(steps + 1)]
    vals = [total(rows, s, now_ts, iv_fallback) for s in grid]
    flips = []
    for a, b, va, vb in zip(grid, grid[1:], vals, vals[1:]):
        if va == 0:
            flips.append(a)
        elif va * vb < 0:
            flips.append(a + (b - a) * va / (va - vb))
    return min(flips, key=lambda x: abs(x - spot)) if flips else None


def summary(rows, spot: float, now_ts: int, iv_fallback: float | None = None) -> dict:
    cells = by_strike(rows, spot, now_ts, iv_fallback)
    if not cells:
        return {"net": None, "call_wall": None, "put_wall": None, "zero_gamma": None, "regime": None}
    net = sum(c["net"] for c in cells.values())
    call_wall = max(cells, key=lambda k: cells[k]["call"])
    put_wall = min(cells, key=lambda k: cells[k]["put"])
    return {"net": net, "call_wall": call_wall if cells[call_wall]["call"] > 0 else None,
            "put_wall": put_wall if cells[put_wall]["put"] < 0 else None,
            "zero_gamma": zero_gamma(rows, spot, now_ts, iv_fallback),
            "regime": "positive" if net > 0 else "negative"}
