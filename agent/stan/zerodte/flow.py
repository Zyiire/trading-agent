"""Put/call volume: how much traders are buying puts against calls, per expiry and through the day.

TZ's put/call table in the reel is volume by expiration. These are the same numbers from the free
chain snapshots: contract volume, and premium (volume x price x 100) so a pile of cheap far-out puts
doesn't outweigh real money near the price.

What the free data can't do: the $200K "sweep" filter needs every option trade with its exchange
and timing. That needs a trade-level feed (ThetaData), so it waits until the free tests earn it.
"""
from __future__ import annotations

from datetime import date


def put_call(rows, expiry: date | None = None) -> dict:
    """Volume and premium totals for one expiry (or all), with put/call ratios (None if no calls)."""
    cv = pv = cp = pp = 0.0
    for r in rows:
        if expiry is not None and r.expiry != expiry:
            continue
        v = r.volume or 0.0
        px = r.mid or 0.0
        if r.type == "call":
            cv, cp = cv + v, cp + v * px * 100
        else:
            pv, pp = pv + v, pp + v * px * 100
    return {"call_volume": cv, "put_volume": pv, "ratio": pv / cv if cv else None,
            "call_premium": round(cp, 2), "put_premium": round(pp, 2), "premium_ratio": pp / cp if cp else None}


def by_expiry(rows) -> list[dict]:
    """TZ's table: one line per expiry, nearest first."""
    return [{"expiry": e, **put_call(rows, e)} for e in sorted({r.expiry for r in rows})]


def intraday(store, symbol: str, day: date) -> list[dict]:
    """The 0DTE put/call ratio at every snapshot saved on `day` (volume is cumulative for the day)."""
    from agent.stan.zerodte.bars import day_bounds
    from agent.stan.zerodte.chain import rows_from_store

    start, end = day_bounds(day)
    out = []
    for ts in store.snapshot_times(symbol, start - 3600, end + 3600):
        rows = rows_from_store(store.chain_at(symbol, ts))
        out.append({"ts": ts, **put_call(rows, day)})
    return out
