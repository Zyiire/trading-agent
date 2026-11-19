"""The box-breakout rules as pure functions, shared by the exam and live trading.

All times are New York. Bars are built from 1-minute bars; nothing here looks at a bar that
has not finished by the moment the decision is made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ET = "America/New_York"
TARGET_R = 3.0
FLAT = time(15, 55)          # exit at the close of the bar that starts here
BOX_START, BOX_END = 9 * 60, 9 * 60 + 25          # five-minute bars opening 9:00 .. 9:25
SIGNAL_START, SIGNAL_END = 9 * 60 + 30, 10 * 60 + 25   # bars opening 9:30 .. 10:25 (close by 10:30)
OHLC = {"o": "first", "h": "max", "l": "min", "c": "last"}


def minute_frame(minute_rows) -> pd.DataFrame:
    """ET-indexed o/h/l/c frame from a DataFrame, or from rows of (ts, o, h, l, c, ...)."""
    if isinstance(minute_rows, pd.DataFrame):
        return minute_rows[["o", "h", "l", "c"]]
    rows = list(minute_rows)
    if not rows:
        return pd.DataFrame(columns=["o", "h", "l", "c"], index=pd.DatetimeIndex([], tz=ET))
    df = pd.DataFrame([r[:5] for r in rows], columns=["ts", "o", "h", "l", "c"]).sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(ET)
    return df[["o", "h", "l", "c"]].astype(float)


def bars(minute_rows, minutes: int) -> pd.DataFrame:
    """`minutes`-long bars (label = bar open) from 1-minute bars. Buckets with no minutes are dropped."""
    m = minute_frame(minute_rows)
    return m.resample(f"{minutes}min", label="left", closed="left").agg(OHLC).dropna()


def _minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.hour * 60 + index.minute)


def box(five_day: pd.DataFrame) -> tuple[float, float] | None:
    """(high, low) of the six 5-minute bars opening 9:00-9:25, or None if any of the six is missing."""
    if five_day.empty:
        return None
    mod = _minute_of_day(five_day.index)
    pre = five_day[(mod >= BOX_START) & (mod <= BOX_END)]
    if len(pre) < 6:
        return None
    return float(pre["h"].max()), float(pre["l"].min())


def trend_flags(fifteen: pd.DataFrame) -> pd.Series:
    """True where the 15-minute EMA(50) is above EMA(200), indexed by bar open."""
    e50 = fifteen["c"].ewm(span=50, adjust=False).mean()
    e200 = fifteen["c"].ewm(span=200, adjust=False).mean()
    return e50 > e200


def trend_up(fifteen: pd.DataFrame, at, flags: pd.Series | None = None) -> bool | None:
    """Trend direction as of `at`, using only 15-minute bars that have finished by then.

    True = EMA(50) above EMA(200), False = below, None = no finished bar yet. Pass `flags`
    (from trend_flags) to skip recomputing the EMAs on every call.
    """
    at = pd.Timestamp(at)
    cutoff = at - pd.Timedelta(minutes=15)
    if flags is None:
        done = fifteen[fifteen.index <= cutoff]
        if done.empty:
            return None
        return bool(trend_flags(done).iloc[-1])
    i = flags.index.searchsorted(cutoff, side="right") - 1
    return None if i < 0 else bool(flags.iloc[i])


@dataclass(frozen=True)
class Signal:
    side: str            # "long" or "short"
    bar_open: pd.Timestamp
    entry: float         # the breakout bar's close
    stop: float          # the opposite side of the box
    target: float
    risk: float          # 1R, in dollars a share


def signal(five_day: pd.DataFrame, box_: tuple[float, float],
           trend: Callable[[pd.Timestamp], bool | None] | None = None,
           target_r: float = TARGET_R) -> Signal | None:
    """The day's first 5-minute bar (opening 9:30-10:25) that closes outside the box.

    `trend` is None for no filter, or a function taking the breakout bar's close time and
    returning True (up), False (down) or None (unknown, so no trade). A long needs up, a short down.
    """
    hi, lo = box_
    mod = _minute_of_day(five_day.index)
    window = five_day[(mod >= SIGNAL_START) & (mod <= SIGNAL_END)]
    if window.empty:
        return None
    close = window["c"].to_numpy()
    outside = (close > hi) | (close < lo)
    if not outside.any():
        return None
    i = int(np.argmax(outside))
    side = "long" if close[i] > hi else "short"
    opened = window.index[i]
    if trend is not None:
        up = trend(opened + pd.Timedelta(minutes=5))
        if up is None or (side == "long") != up:
            return None
    entry = float(close[i])
    stop = lo if side == "long" else hi
    risk = abs(entry - stop)
    target = entry + target_r * risk if side == "long" else entry - target_r * risk
    return Signal(side, opened, entry, float(stop), float(target), float(risk))


def simulate_exit(minutes_after: pd.DataFrame, sig: Signal, flat: time = FLAT) -> tuple[float, str] | None:
    """Walk 1-minute bars after the entry. Returns (exit_price, "stop" | "target" | "time").

    A bar touching both stop and target counts as the stop. Bars after `flat` are ignored; the
    time exit is the close of the last bar at or before `flat`. None if there are no such bars.
    """
    mod = _minute_of_day(minutes_after.index)
    win = minutes_after[mod <= flat.hour * 60 + flat.minute]
    if win.empty:
        return None
    hi, lo = win["h"].to_numpy(), win["l"].to_numpy()
    if sig.side == "long":
        stop_hit, target_hit = lo <= sig.stop, hi >= sig.target
    else:
        stop_hit, target_hit = hi >= sig.stop, lo <= sig.target
    s = int(np.argmax(stop_hit)) if stop_hit.any() else None
    t = int(np.argmax(target_hit)) if target_hit.any() else None
    if s is not None and (t is None or s <= t):
        return sig.stop, "stop"
    if t is not None:
        return sig.target, "target"
    return float(win["c"].iloc[-1]), "time"


def size(equity: float, risk_per_share: float, buying_power: float, risk_pct: float = 0.01,
         price: float | None = None) -> int:
    """Whole shares risking `risk_pct` of equity. 0 means skip.

    Skips (0) when under one share, or when `price` is given and the position would cost more
    than 95% of buying power.
    """
    if equity <= 0 or risk_per_share <= 0:
        return 0
    qty = math.floor(equity * risk_pct / risk_per_share)
    if qty < 1:
        return 0
    if price is not None and qty * price > 0.95 * buying_power:
        return 0
    return qty
