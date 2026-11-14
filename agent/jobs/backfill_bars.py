"""Step 2: download daily price history for the watchlist into the database.

Run:  python -m agent.jobs.backfill_bars
Safe to re-run: it only fetches days it does not have yet.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from typing import Sequence

from agent.broker.base import Broker
from agent.config import ConfigError, load_settings
from agent.store import Store

# Alpaca's free IEX feed starts around 2016, so the request asks from 2011
# and takes whatever history is available.
HISTORY_START = date(2011, 1, 1)


def backfill(broker: Broker, store: Store, symbols: Sequence[str], today: date | None = None) -> dict[str, int]:
    today = today or date.today()
    saved: dict[str, int] = {}
    for symbol in symbols:
        last = store.last_bar_day(symbol)
        start = last + timedelta(days=1) if last else HISTORY_START
        if start > today:
            saved[symbol] = 0
            continue
        bars = broker.get_daily_bars([symbol], start=start)
        saved[symbol] = store.save_daily_bars(bars)
    return saved


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as err:
        print(f"Setup problem: {err}")
        return 1

    from agent.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker(settings)
    store = Store(settings.db_path)
    try:
        saved = backfill(broker, store, settings.watchlist)
    except Exception as err:
        print(f"Download failed: {type(err).__name__}: {err}")
        return 1
    for symbol, n in saved.items():
        print(f"  {symbol}: {n} new days (total {store.count_bars(symbol)}, through {store.last_bar_day(symbol)})")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
