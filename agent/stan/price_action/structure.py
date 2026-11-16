"""1. Market structure: confirmed swing pivots on the higher timeframe, and the trend they spell out.

A swing high at HTF bar j is a high strictly above the N bars to its left and at least as high as
the N bars to its right. It cannot be known until bar j+N has closed, so each pivot carries
`confirmed_at`: the LTF index of the close of HTF bar j+N. Nothing before that bar may use it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.stan.price_action.series import HTF

UP, DOWN, CHOP = 1, -1, 0


@dataclass(frozen=True)
class Pivot:
    kind: str           # "high" or "low"
    price: float
    htf_index: int
    confirmed_at: int   # LTF bar index from which this pivot may be used


def find_pivots(htf: HTF, n: int) -> list[Pivot]:
    """All swing highs and lows, sorted by when they become known."""
    out = []
    for j in range(n, len(htf.h) - n):
        left_h, right_h = htf.h[j - n:j], htf.h[j + 1:j + n + 1]
        left_l, right_l = htf.l[j - n:j], htf.l[j + 1:j + n + 1]
        when = int(htf.close_idx[j + n])
        if htf.h[j] > left_h.max() and htf.h[j] >= right_h.max():
            out.append(Pivot("high", float(htf.h[j]), j, when))
        if htf.l[j] < left_l.min() and htf.l[j] <= right_l.min():
            out.append(Pivot("low", float(htf.l[j]), j, when))
    out.sort(key=lambda p: (p.confirmed_at, p.htf_index))
    return out


def classify(highs: list[float], lows: list[float], k: int) -> int:
    """UP for higher highs and higher lows, DOWN for lower highs and lower lows, else CHOP."""
    if len(highs) < k or len(lows) < k:
        return CHOP
    hs, ls = highs[-k:], lows[-k:]
    if all(b > a for a, b in zip(hs, hs[1:])) and all(b > a for a, b in zip(ls, ls[1:])):
        return UP
    if all(b < a for a, b in zip(hs, hs[1:])) and all(b < a for a, b in zip(ls, ls[1:])):
        return DOWN
    return CHOP


def trend_by_bar(pivots: list[Pivot], n_ltf: int, k: int) -> np.ndarray:
    """Trend known at the close of every LTF bar, using only pivots confirmed by then."""
    trend = np.zeros(n_ltf, int)
    highs, lows, p, cur = [], [], 0, CHOP
    for i in range(n_ltf):
        while p < len(pivots) and pivots[p].confirmed_at <= i:
            (highs if pivots[p].kind == "high" else lows).append(pivots[p].price)
            cur = classify(highs, lows, k)
            p += 1
        trend[i] = cur
    return trend


def known_pivots(pivots: list[Pivot], i: int, kind: str, count: int) -> list[float]:
    """The most recent `count` pivots of a kind that are confirmed at LTF bar i."""
    out = [pv.price for pv in pivots if pv.kind == kind and pv.confirmed_at <= i]
    return out[-count:]


def ltf_swing(low_or_high: np.ndarray, i: int, lookback: int, lowest: bool) -> float:
    """The local LTF swing extreme over the last `lookback` closed bars, for a structural stop."""
    w = low_or_high[max(0, i - lookback + 1):i + 1]
    return float(w.min() if lowest else w.max())
