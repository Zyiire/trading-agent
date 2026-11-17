"""One SQLite file for the 0DTE bot's data. Every save is an upsert, so re-running a job is safe."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1m (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,          -- bar start, Unix seconds (UTC)
    o REAL, h REAL, l REAL, c REAL, v REAL, vwap REAL, source TEXT,
    PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS chain (
    taken_at INTEGER NOT NULL, symbol TEXT NOT NULL, expiry TEXT NOT NULL, strike REAL NOT NULL,
    type TEXT NOT NULL, bid REAL, ask REAL, last REAL, iv REAL, delta REAL, gamma REAL,
    volume REAL, spot REAL, source TEXT,
    PRIMARY KEY (taken_at, symbol, expiry, strike, type));
CREATE INDEX IF NOT EXISTS idx_chain_day ON chain (symbol, taken_at);
CREATE TABLE IF NOT EXISTS open_interest (
    day TEXT NOT NULL, symbol TEXT NOT NULL, expiry TEXT NOT NULL, strike REAL NOT NULL,
    type TEXT NOT NULL, oi REAL, source TEXT,
    PRIMARY KEY (day, symbol, expiry, strike, type));
"""

CHAIN_COLS = ("taken_at", "symbol", "expiry", "strike", "type", "bid", "ask", "last", "iv", "delta",
              "gamma", "volume", "spot", "source")


class ZeroStore:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ----- bars -----
    def save_bars(self, symbol: str, rows: Iterable[tuple], source: str) -> int:
        """rows: (ts, o, h, l, c, v, vwap)."""
        data = [(symbol, int(r[0]), *map(_num, r[1:7]), source) for r in rows]
        self.db.executemany("INSERT OR REPLACE INTO bars_1m VALUES (?,?,?,?,?,?,?,?,?)", data)
        self.db.commit()
        return len(data)

    def bars(self, symbol: str, start_ts: int | None = None, end_ts: int | None = None) -> list[tuple]:
        q, args = "SELECT ts,o,h,l,c,v,vwap FROM bars_1m WHERE symbol=?", [symbol]
        if start_ts is not None:
            q, args = q + " AND ts>=?", args + [start_ts]
        if end_ts is not None:
            q, args = q + " AND ts<?", args + [end_ts]
        return self.db.execute(q + " ORDER BY ts", args).fetchall()

    def bar_span(self, symbol: str, source: str | None = None) -> tuple[int, int, int] | None:
        """(first ts, last ts, count), or None when there are no bars. `source` limits it to one feed."""
        q, args = "SELECT MIN(ts), MAX(ts), COUNT(*) FROM bars_1m WHERE symbol=?", [symbol]
        if source:
            q, args = q + " AND source=?", args + [source]
        first, last, n = self.db.execute(q, args).fetchone()
        return None if not n else (first, last, n)

    # ----- chain snapshots -----
    def save_chain(self, taken_at: int, symbol: str, rows: Iterable, spot: float | None, source: str) -> int:
        data = [(int(taken_at), symbol, r.expiry.isoformat(), float(r.strike), r.type, _num(r.bid), _num(r.ask),
                 _num(r.last), _num(r.iv), _num(r.delta), _num(r.gamma), _num(r.volume), _num(spot), source)
                for r in rows]
        self.db.executemany(f"INSERT OR REPLACE INTO chain VALUES ({','.join('?' * len(CHAIN_COLS))})", data)
        self.db.commit()
        return len(data)

    def snapshot_times(self, symbol: str, start_ts: int | None = None, end_ts: int | None = None) -> list[int]:
        q, args = "SELECT DISTINCT taken_at FROM chain WHERE symbol=?", [symbol]
        if start_ts is not None:
            q, args = q + " AND taken_at>=?", args + [start_ts]
        if end_ts is not None:
            q, args = q + " AND taken_at<?", args + [end_ts]
        return [r[0] for r in self.db.execute(q + " ORDER BY taken_at", args)]

    def chain_at(self, symbol: str, taken_at: int) -> list[dict]:
        cur = self.db.execute(f"SELECT {','.join(CHAIN_COLS)} FROM chain WHERE symbol=? AND taken_at=?",
                              (symbol, taken_at))
        return [dict(zip(CHAIN_COLS, r)) for r in cur]

    # ----- open interest -----
    def save_open_interest(self, day: date, symbol: str, rows: Iterable[tuple], source: str) -> int:
        """rows: (expiry, strike, type, oi)."""
        data = [(day.isoformat(), symbol, e.isoformat(), float(k), t, _num(oi), source) for e, k, t, oi in rows]
        self.db.executemany("INSERT OR REPLACE INTO open_interest VALUES (?,?,?,?,?,?,?)", data)
        self.db.commit()
        return len(data)

    def open_interest(self, day: date, symbol: str) -> dict[tuple[str, float, str], float]:
        cur = self.db.execute("SELECT expiry, strike, type, oi FROM open_interest WHERE day=? AND symbol=?",
                              (day.isoformat(), symbol))
        return {(e, k, t): oi for e, k, t, oi in cur if oi is not None}

    def oi_days(self, symbol: str) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT day FROM open_interest WHERE symbol=? ORDER BY day", (symbol,))]

    def snapshot_days(self, symbol: str) -> dict[str, int]:
        """Snapshots per New York day."""
        from agent.stan.zerodte.bars import et_day
        out: dict[str, int] = {}
        for ts in self.snapshot_times(symbol):
            d = et_day(ts).isoformat()
            out[d] = out.get(d, 0) + 1
        return out


def _num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if x != x else x  # NaN becomes NULL
