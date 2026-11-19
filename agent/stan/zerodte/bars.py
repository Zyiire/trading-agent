"""QQQ bars: 1-minute bars in, 3-minute regular-session bars with session VWAP out.

Sources (both free):
  Alpaca historical  full-market (SIP) 1-minute bars back to 2016, except the latest 15 minutes,
                     which the free plan doesn't serve. Used for the backfill.
  Alpaca IEX         live 1-minute bars from the IEX exchange only. Prices track the market closely;
                     volume is a small slice of the total, so VWAP from it is approximate.
  Yahoo              about a week of 1-minute bars with full volume, a few seconds behind. The live backup.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN_MIN, CLOSE_MIN = 9 * 60 + 30, 16 * 60  # minutes after midnight, New York time
FREE_SIP_DELAY = timedelta(minutes=16)


def et_time(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET)


def et_day(ts: int) -> date:
    return et_time(ts).date()


def day_bounds(day: date) -> tuple[int, int]:
    """Unix seconds of 9:30 AM and 4:00 PM New York time on `day`."""
    start = datetime.combine(day, time(9, 30), tzinfo=ET)
    return int(start.timestamp()), int(start.timestamp()) + 390 * 60


def session_bars(rows: Iterable[tuple], minutes: int = 3, now_ts: int | None = None) -> dict[date, list[dict]]:
    """Group 1-minute bars (ts, o, h, l, c, v, vwap) into `minutes`-long regular-session bars per day.

    Each bar: slot (0 = the 9:30 bar), ts (its start), o h l c v, and vwap = the session's VWAP at the
    bar's close. A bucket that hasn't finished by `now_ts` is left out, so live code never acts on a
    bar that is still forming. Pre- and after-market minutes are dropped.
    """
    days: dict[date, dict[int, dict]] = {}
    for ts, o, h, l, c, v, vw in rows:
        t = et_time(int(ts))
        m = t.hour * 60 + t.minute
        if not OPEN_MIN <= m < CLOSE_MIN:
            continue
        slot = (m - OPEN_MIN) // minutes
        start = int(ts) - (m - OPEN_MIN - slot * minutes) * 60 - t.second
        if now_ts is not None and start + minutes * 60 > now_ts:
            continue
        v = float(v or 0)
        price = float(vw) if vw else (float(h) + float(l) + float(c)) / 3
        b = days.setdefault(t.date(), {}).get(slot)
        if b is None:
            days[t.date()][slot] = b = {"slot": slot, "ts": start, "o": float(o), "h": float(h), "l": float(l),
                                        "c": float(c), "v": 0.0, "pv": 0.0, "_last": int(ts)}
        else:
            b["h"], b["l"] = max(b["h"], float(h)), min(b["l"], float(l))
            if int(ts) >= b["_last"]:
                b["c"], b["_last"] = float(c), int(ts)
        b["v"] += v
        b["pv"] += price * v
    out: dict[date, list[dict]] = {}
    for d in sorted(days):
        cum_v = cum_pv = 0.0
        bars = []
        for slot in sorted(days[d]):
            b = days[d][slot]
            cum_v, cum_pv = cum_v + b["v"], cum_pv + b["pv"]
            bars.append({"slot": b["slot"], "ts": b["ts"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"],
                         "v": b["v"], "vwap": cum_pv / cum_v if cum_v else b["c"]})
        out[d] = bars
    return out


# ----- fetching -----
def fetch_alpaca(client, symbol: str, start: datetime, end: datetime, feed: str = "sip") -> list[tuple]:
    """1-minute bars from Alpaca, a month per request."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    rows: list[tuple] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(days=31), end)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=cursor, end=stop,
                               adjustment=Adjustment.SPLIT, feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
        data = client.get_stock_bars(req).data.get(symbol, [])
        rows += [(int(b.timestamp.timestamp()), b.open, b.high, b.low, b.close, b.volume, b.vwap) for b in data]
        cursor = stop
    return rows


def fetch_yahoo(symbol: str, period: str = "7d", prepost: bool = False) -> list[tuple]:
    """The last ~7 days (`period`) of 1-minute bars from Yahoo; `prepost` adds pre- and after-market minutes.
    The newest row can be the current, unfinished minute."""
    import yfinance as yf

    df = yf.download(symbol, period=period, interval="1m", auto_adjust=False, progress=False, prepost=prepost)
    return frame_rows(df)


def frame_rows(df) -> list[tuple]:
    if df is None or not len(df):
        return []
    if getattr(df.columns, "nlevels", 1) > 1:
        df = df.droplevel(1, axis=1)
    out = []
    for ts, r in df.iterrows():
        stamp = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        out.append((int(stamp.timestamp()), float(r["Open"]), float(r["High"]), float(r["Low"]),
                    float(r["Close"]), float(r.get("Volume", 0) or 0), None))
    return out


def backfill(store, client, symbol: str, since: date, now: datetime | None = None, log=print) -> int:
    """Bring the stored 1-minute history up to 16 minutes ago. Picks up where it left off."""
    now = now or datetime.now(timezone.utc)
    span = store.bar_span(symbol, source="alpaca-sip")  # live IEX/Yahoo bars get replaced
    start = (datetime.fromtimestamp(span[1] + 60, tz=timezone.utc) if span
             else datetime.combine(since, time.min, tzinfo=timezone.utc))
    end = now - FREE_SIP_DELAY
    total = 0
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(days=92), end)
        rows = fetch_alpaca(client, symbol, cursor, stop, feed="sip")
        total += store.save_bars(symbol, rows, "alpaca-sip")
        log(f"  {symbol} bars {cursor:%Y-%m-%d} to {stop:%Y-%m-%d}: {len(rows):,}")
        cursor = stop
    return total
