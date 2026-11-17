"""0DTE QQQ options bot (tz.fund-style), built in phases.

Phase 2 (this package so far) is the data pipeline, free sources only:
  bars.py      QQQ 1-minute bars from Alpaca, rolled into 3-minute session bars with VWAP
  chain.py     option chain snapshots: quotes, IV, greeks, volume (Alpaca), open interest
               (Alpaca contracts, Yahoo as a backup)
  flow.py      put/call volume and premium ratios, per expiry and through the day
  gex.py       gamma exposure by strike, call and put walls, the zero-gamma level
  levels.py    the day's key levels: round numbers, prior-day high/low/close, GEX walls, big OI strikes
  calendar.py  CPI, jobs report and FOMC days (official BLS and Fed dates, 2021 to 2027)
  store.py     one SQLite file (data/zerodte.db) for all of it
  collect.py   the command line: backfill, snapshot, watch, levels, status

No signals and no trading here yet: those are Phases 3 and later.
"""
