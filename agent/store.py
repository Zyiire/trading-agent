"""SQLite storage: price history, options chain snapshots, account snapshots.

SQLite keeps setup at zero. The spec's Postgres can replace it later behind
this same class.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from agent.broker.base import AccountInfo, DailyBar, OptionQuote

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol TEXT NOT NULL,
    day TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume REAL NOT NULL, vwap REAL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS option_snapshots (
    taken_at TEXT NOT NULL,
    contract TEXT NOT NULL,
    underlying TEXT NOT NULL,
    underlying_price REAL,
    expiration TEXT NOT NULL,
    option_type TEXT NOT NULL,
    strike REAL NOT NULL,
    bid REAL, ask REAL, bid_size REAL, ask_size REAL, last_price REAL,
    implied_volatility REAL, delta REAL, gamma REAL, theta REAL, vega REAL,
    PRIMARY KEY (taken_at, contract)
);
CREATE INDEX IF NOT EXISTS idx_option_snapshots_underlying
    ON option_snapshots (underlying, taken_at);
CREATE TABLE IF NOT EXISTS account_snapshots (
    taken_at TEXT PRIMARY KEY,
    equity REAL NOT NULL, cash REAL NOT NULL, buying_power REAL NOT NULL,
    is_paper INTEGER NOT NULL
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # ----- daily bars -----
    def save_daily_bars(self, bars: Iterable[DailyBar]) -> int:
        rows = [
            (b.symbol, b.day.isoformat(), b.open, b.high, b.low, b.close, b.volume, b.vwap)
            for b in bars
        ]
        self._db.executemany(
            "INSERT OR REPLACE INTO daily_bars VALUES (?,?,?,?,?,?,?,?)", rows
        )
        self._db.commit()
        return len(rows)

    def last_bar_day(self, symbol: str) -> date | None:
        row = self._db.execute(
            "SELECT MAX(day) FROM daily_bars WHERE symbol = ?", (symbol,)
        ).fetchone()
        return date.fromisoformat(row[0]) if row and row[0] else None

    def last_close(self, symbol: str) -> float | None:
        row = self._db.execute(
            "SELECT close FROM daily_bars WHERE symbol = ? ORDER BY day DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return float(row[0]) if row else None

    def count_bars(self, symbol: str | None = None) -> int:
        if symbol:
            row = self._db.execute(
                "SELECT COUNT(*) FROM daily_bars WHERE symbol = ?", (symbol,)
            ).fetchone()
        else:
            row = self._db.execute("SELECT COUNT(*) FROM daily_bars").fetchone()
        return int(row[0])

    # ----- option snapshots -----
    def save_option_snapshot(
        self,
        quotes: Iterable[OptionQuote],
        underlying_price: float | None,
        taken_at: str | None = None,
    ) -> int:
        taken_at = taken_at or utc_now_iso()
        rows = [
            (
                taken_at, q.contract, q.underlying, underlying_price,
                q.expiration.isoformat(), q.option_type, q.strike,
                q.bid, q.ask, q.bid_size, q.ask_size, q.last_price,
                q.implied_volatility, q.delta, q.gamma, q.theta, q.vega,
            )
            for q in quotes
        ]
        self._db.executemany(
            "INSERT OR REPLACE INTO option_snapshots VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._db.commit()
        return len(rows)

    def count_option_rows(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM option_snapshots").fetchone()[0])

    # ----- account -----
    def save_account_snapshot(self, account: AccountInfo, taken_at: str | None = None) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO account_snapshots VALUES (?,?,?,?,?)",
            (
                taken_at or utc_now_iso(), account.equity, account.cash,
                account.buying_power, int(account.is_paper),
            ),
        )
        self._db.commit()
