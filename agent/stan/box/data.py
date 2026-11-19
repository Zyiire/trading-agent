"""QQQ 1-minute bars (Alpaca SIP) in data/box_bars.db, backfilled and topped up.

    python -m agent.stan.box.data backfill [--since 2016-01-01] [--db data/box_bars.db]
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from agent.stan.box.rules import ET
from agent.stan.zerodte.bars import fetch_alpaca

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "box_bars.db"
SYMBOL = "QQQ"
SOURCE = "alpaca-sip"
SIP_DELAY = timedelta(minutes=20)   # newer than this is off-limits on the free SIP plan
SCHEMA = """CREATE TABLE IF NOT EXISTS bars_1m (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,          -- bar start, Unix seconds (UTC)
    o REAL, h REAL, l REAL, c REAL, v REAL, vwap REAL, source TEXT,
    PRIMARY KEY (symbol, ts))"""


def load(path: str | Path = DB_PATH, symbol: str = SYMBOL) -> pd.DataFrame:
    """All stored 1-minute bars as an ET-indexed o/h/l/c frame (read-only)."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        df = pd.read_sql("SELECT ts, o, h, l, c FROM bars_1m WHERE symbol=? ORDER BY ts", db, params=(symbol,))
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(ET)
    return df[["o", "h", "l", "c"]]


def backfill(client, path: str | Path = DB_PATH, symbol: str = SYMBOL, since: date = date(2016, 1, 1),
             now: datetime | None = None, log=print) -> int:
    """Fetch SIP bars from the last stored bar (or `since`) up to 20 minutes ago. Safe to re-run."""
    now = now or datetime.now(timezone.utc)
    end = now - SIP_DELAY
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    try:
        db.execute(SCHEMA)
        last = db.execute("SELECT MAX(ts) FROM bars_1m WHERE symbol=?", (symbol,)).fetchone()[0]
        cursor = (datetime.fromtimestamp(last + 60, tz=timezone.utc) if last is not None
                  else datetime.combine(since, time.min, tzinfo=timezone.utc))
        total = 0
        while cursor < end:
            stop = min(cursor + timedelta(days=92), end)
            rows = fetch_alpaca(client, symbol, cursor, stop, feed="sip")
            db.executemany("INSERT OR REPLACE INTO bars_1m VALUES (?,?,?,?,?,?,?,?,?)",
                           [(symbol, int(r[0]), *(None if x is None else float(x) for x in r[1:7]), SOURCE)
                            for r in rows])
            db.commit()
            total += len(rows)
            log(f"  {symbol} bars {cursor:%Y-%m-%d} to {stop:%Y-%m-%d}: {len(rows):,}")
            cursor = stop
        return total
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent.stan.box.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    bf = sub.add_parser("backfill", help="fetch QQQ SIP 1-minute bars, resuming from the last stored bar")
    bf.add_argument("--since", type=date.fromisoformat, default=date(2016, 1, 1))
    bf.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args(argv)
    from alpaca.data.historical import StockHistoricalDataClient
    from agent.config import load_settings
    s = load_settings()
    client = StockHistoricalDataClient(s.api_key, s.secret_key)
    n = backfill(client, args.db, since=args.since)
    print(f"Saved {n:,} new bars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
