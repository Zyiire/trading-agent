"""Bar data as flat arrays, and the higher-timeframe bars built from them.

Lookahead rule used everywhere in this package: a value "available at LTF bar i" may use only
bars 0..i, all of which have closed when bar i closes. Each HTF bar records `close_idx`, the LTF
index of its last 5-minute bar; an HTF bar is usable at LTF bar i only when close_idx <= i.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Series:
    """One symbol's complete regular sessions back to back (5-minute bars)."""
    symbol: str
    days: list[str]          # session date per session
    day_idx: np.ndarray      # session number per bar
    slot: np.ndarray         # 0..77 within the session
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    premarket: dict[str, tuple[float, float]]  # date -> (high, low), when the data has premarket bars

    def __len__(self) -> int:
        return len(self.c)

    def head(self, n: int) -> "Series":
        """The first n bars only: what was known when bar n-1 closed (used by the lookahead tests)."""
        k = int(self.day_idx[n - 1]) + 1 if n else 0
        return Series(self.symbol, self.days[:k], self.day_idx[:n], self.slot[:n], self.o[:n], self.h[:n],
                      self.l[:n], self.c[:n], self.v[:n], {d: self.premarket[d] for d in self.days[:k]
                                                           if d in self.premarket})


def from_sessions(symbol: str, sessions: list[tuple[str, list[dict]]],
                  premarket: dict[str, tuple[float, float]] | None = None) -> Series:
    days, rows = [], []
    for k, (d, bars) in enumerate(sessions):
        days.append(d)
        for i, b in enumerate(bars):
            rows.append((k, b.get("slot", i), b["o"], b["h"], b["l"], b["c"], b.get("v") or 0))
    a = np.array(rows, float) if rows else np.zeros((0, 7))
    return Series(symbol, days, a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3], a[:, 4], a[:, 5],
                  a[:, 6], dict(premarket or {}))


def atr(s: Series, period: int) -> np.ndarray:
    """Average true range ending at each bar (uses bars up to and including that bar)."""
    prev = np.concatenate([[s.c[0]], s.c[:-1]]) if len(s) else s.c
    tr = np.maximum(s.h - s.l, np.maximum(np.abs(s.h - prev), np.abs(s.l - prev)))
    out = np.empty_like(tr)
    run = 0.0
    for i, x in enumerate(tr):
        run += x
        if i >= period:
            run -= tr[i - period]
        out[i] = run / min(i + 1, period)
    return out


def prior_mean(x: np.ndarray, period: int) -> np.ndarray:
    """Mean of the `period` bars BEFORE each bar (the bar itself excluded); nan until there are enough."""
    out = np.full(len(x), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    for i in range(period, len(x)):
        out[i] = (cs[i] - cs[i - period]) / period
    return out


@dataclass
class HTF:
    h: np.ndarray
    l: np.ndarray
    close_idx: np.ndarray    # LTF index of the HTF bar's last 5-minute bar


def build_htf(s: Series, bars_per: int, session_bars: int = 78) -> HTF:
    """Group each session's 5-minute bars into blocks of `bars_per`. Blocks never span sessions; a
    short block at the end of a session closes with the session. A block whose last 5-minute bar
    hasn't closed yet (the HTF bar still forming at the end of the data) is left out entirely, so
    a half-built HTF bar can never confirm a pivot."""
    hs, ls, ci = [], [], []
    i, n = 0, len(s)
    while i < n:
        d, block = s.day_idx[i], s.slot[i] // bars_per
        j = i
        while j + 1 < n and s.day_idx[j + 1] == d and (s.slot[j + 1] // bars_per) == block:
            j += 1
        if s.slot[j] == min((block + 1) * bars_per, session_bars) - 1:
            hs.append(s.h[i:j + 1].max())
            ls.append(s.l[i:j + 1].min())
            ci.append(j)
        i = j + 1
    return HTF(np.array(hs), np.array(ls), np.array(ci, int))
