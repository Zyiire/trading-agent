"""Step 3: save options chain snapshots every 15 minutes while the market is open.

Run:  python -m agent.jobs.snapshot_chains          (keeps running)
      python -m agent.jobs.snapshot_chains --once   (one snapshot, then exit)

Every day this runs adds to your own free options history, which the
backtester and the paper-trading fill simulator will use later.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from typing import Sequence

from agent.broker.base import Broker
from agent.config import ConfigError, Settings, load_settings
from agent.store import Store, utc_now_iso


def snapshot_once(
    broker: Broker,
    store: Store,
    symbols: Sequence[str],
    settings: Settings,
    today: date | None = None,
    taken_at: str | None = None,
) -> dict[str, int]:
    """Save one snapshot per symbol. One symbol failing does not stop the rest."""
    today = today or date.today()
    taken_at = taken_at or utc_now_iso()
    saved: dict[str, int] = {}
    for symbol in symbols:
        price = store.last_close(symbol)
        strike_min = strike_max = None
        if price:
            strike_min = round(price * (1 - settings.snapshot_strike_band), 2)
            strike_max = round(price * (1 + settings.snapshot_strike_band), 2)
        try:
            quotes = broker.get_option_chain(
                symbol,
                expiration_from=today,
                expiration_to=today + timedelta(days=settings.snapshot_max_days_to_expiry),
                strike_min=strike_min,
                strike_max=strike_max,
            )
            saved[symbol] = store.save_option_snapshot(quotes, price, taken_at)
        except Exception as err:
            print(f"  {symbol}: snapshot failed ({type(err).__name__}: {err})")
            saved[symbol] = -1
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save options chain snapshots.")
    parser.add_argument("--once", action="store_true", help="take one snapshot and exit")
    parser.add_argument("--even-if-closed", action="store_true", help="snapshot even when the market is closed")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as err:
        print(f"Setup problem: {err}")
        return 1

    from agent.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker(settings)
    store = Store(settings.db_path)
    interval = settings.snapshot_interval_minutes * 60

    while True:
        try:
            is_open = broker.get_clock().is_open
        except Exception as err:
            print(f"Could not check the market clock ({type(err).__name__}); retrying later.")
            is_open = False
        if is_open or args.even_if_closed:
            saved = snapshot_once(broker, store, settings.watchlist, settings)
            ok = {s: n for s, n in saved.items() if n >= 0}
            print(f"{utc_now_iso()} saved {sum(ok.values())} contracts across {len(ok)} symbols "
                  f"(database total {store.count_option_rows()})")
        else:
            print(f"{utc_now_iso()} market closed, waiting")
        if args.once:
            break
        time.sleep(interval)

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
