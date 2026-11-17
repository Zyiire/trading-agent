"""The 0DTE data pipeline's command line.

  python -m agent.stan.zerodte.collect backfill   1-minute QQQ history since 2021 (resumes; first run is long)
  python -m agent.stan.zerodte.collect snapshot   one chain + open interest + bars snapshot now
  python -m agent.stan.zerodte.collect watch      snapshot every 5 minutes until the close (the scheduled job)
  python -m agent.stan.zerodte.collect levels     today's key levels, GEX and put/call table
  python -m agent.stan.zerodte.collect status     what has been collected so far

Uses the Alpaca paper keys in .env (read-only market data; nothing here places orders). Every
source has a free backup: when Alpaca fails, Yahoo fills in and the row says which one it came from.
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from agent.stan.zerodte import bars as B
from agent.stan.zerodte import calendar as cal
from agent.stan.zerodte import chain as C
from agent.stan.zerodte import flow
from agent.stan.zerodte.config import ZeroDteConfig
from agent.stan.zerodte.levels import key_levels
from agent.stan.zerodte.store import ZeroStore


class AlpacaSources:
    """Free market data: Alpaca first, Yahoo when Alpaca fails."""

    def __init__(self, settings) -> None:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.trading.client import TradingClient

        if not settings.paper:
            raise RuntimeError("Paper keys only.")
        self.stocks = StockHistoricalDataClient(settings.api_key, settings.secret_key)
        self.options = OptionHistoricalDataClient(settings.api_key, settings.secret_key, raw_data=True)
        self.trading = TradingClient(settings.api_key, settings.secret_key, paper=True)

    def market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def spot(self, symbol: str) -> tuple[float, str]:
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockLatestTradeRequest
            t = self.stocks.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol,
                                                                           feed=DataFeed.IEX))
            return float(t[symbol].price), "alpaca-iex"
        except Exception:
            import yfinance as yf
            return float(yf.Ticker(symbol).fast_info["last_price"]), "yahoo"

    def today_bars(self, symbol: str, day: date) -> tuple[list[tuple], str]:
        start, _ = B.day_bounds(day)
        try:
            rows = B.fetch_alpaca(self.stocks, symbol, datetime.fromtimestamp(start, timezone.utc),
                                  datetime.now(timezone.utc), feed="iex")
            if rows:
                return rows, "alpaca-iex"
        except Exception:
            pass
        return [r for r in B.fetch_yahoo(symbol) if r[0] >= start], "yahoo"

    def chain(self, symbol, today, max_dte, lo, hi) -> tuple[list[C.ChainRow], str]:
        try:
            rows = C.alpaca_chain(self.options, symbol, today, max_dte, lo, hi)
            if rows:
                return rows, "alpaca"
        except Exception:
            pass
        return [r for r in C.yahoo_chain(symbol, today, max_dte) if lo <= r.strike <= hi], "yahoo"

    def open_interest(self, symbol, today, max_dte, lo, hi) -> tuple[list[tuple], str]:
        try:
            rows = C.alpaca_open_interest(self.trading, symbol, today, max_dte, lo, hi)
            if rows:
                return rows, "alpaca"
        except Exception:
            pass
        ych = C.yahoo_chain(symbol, today, max_dte)
        return [(r.expiry, r.strike, r.type, r.oi) for r in ych if r.oi and lo <= r.strike <= hi], "yahoo"


def snapshot(store: ZeroStore, src, cfg: ZeroDteConfig, now: datetime | None = None) -> dict:
    """One pass: today's bars, open interest (once a day), and the option chain."""
    now = now or datetime.now(timezone.utc)
    today, ts, sym = now.astimezone(B.ET).date(), int(now.timestamp()), cfg.symbol
    spot, spot_src = src.spot(sym)
    lo, hi = spot * (1 - cfg.chain_strike_band), spot * (1 + cfg.chain_strike_band)
    out = {"ts": ts, "spot": spot, "spot_source": spot_src}
    try:
        rows, s = src.today_bars(sym, today)
        out["bars"], out["bars_source"] = store.save_bars(sym, rows, s), s
    except Exception as err:
        out["bars_error"] = f"{type(err).__name__}: {err}"
    if not store.open_interest(today, sym):
        try:
            oi, s = src.open_interest(sym, today, cfg.chain_max_dte, lo, hi)
            out["open_interest"], out["oi_source"] = store.save_open_interest(today, sym, oi, s), s
        except Exception as err:
            out["oi_error"] = f"{type(err).__name__}: {err}"
    try:
        rows, s = src.chain(sym, today, cfg.chain_max_dte, lo, hi)
        out["contracts"], out["chain_source"] = store.save_chain(ts, sym, rows, spot, s), s
    except Exception as err:
        out["chain_error"] = f"{type(err).__name__}: {err}"
    return out


def today_view(store: ZeroStore, cfg: ZeroDteConfig, day: date) -> dict | None:
    """Levels, GEX and the put/call table from the latest snapshot saved on `day`."""
    start, end = B.day_bounds(day)
    # Any snapshot saved on `day` in New York time counts, including one taken late at night.
    midnight = datetime.combine(day, datetime.min.time(), tzinfo=B.ET)
    next_midnight = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=B.ET)
    times = store.snapshot_times(cfg.symbol, int(midnight.timestamp()), int(next_midnight.timestamp()))
    if not times:
        return None
    ts = times[-1]
    records = store.chain_at(cfg.symbol, ts)
    spot = next((r["spot"] for r in records if r["spot"]), None)
    rows = C.with_open_interest(C.rows_from_store(records), store.open_interest(day, cfg.symbol))
    sessions = B.session_bars(store.bars(cfg.symbol, start - 5 * 86400, end), cfg.bar_minutes)
    prior = [d for d in sessions if d < day]
    view = key_levels(spot, sessions[prior[-1]] if prior else None, sessions.get(day), rows, ts, cfg)
    view.update({"day": day, "ts": ts, "put_call": flow.by_expiry(rows), "events": cal.events_on(day)})
    return view


def report(view: dict | None, day: date) -> str:
    if view is None:
        return f"No snapshot saved for {day} yet. Run: python -m agent.stan.zerodte.collect snapshot"
    g, lines = view["gex"], []
    ev = view["events"]
    lines.append(f"QQQ {view['spot']:.2f} at {B.et_time(view['ts']):%Y-%m-%d %H:%M} ET")
    lines.append("News day: " + ("unknown (outside the calendar)" if ev is None else ", ".join(ev) or "none"))
    if g["net"] is not None:
        zg = f"{g['zero_gamma']:.2f}" if g["zero_gamma"] else "none nearby"
        lines.append(f"GEX: {g['regime']} (net ${g['net'] / 1e6:+,.0f}M per 1% move), call wall {g['call_wall']}, "
                     f"put wall {g['put_wall']}, zero gamma {zg}")
    else:
        lines.append("GEX: no open interest saved yet")
    lines.append("\nLevels (nearest first):")
    for lv in sorted(view["levels"], key=lambda x: abs(x["price"] - view["spot"]))[:12]:
        lines.append(f"  {lv['price']:>8.2f}  {lv['side']:<5}  strength {lv['strength']:<2}  {', '.join(lv['kinds'])}")
    lines.append("\nPut/call by expiry:")
    for p in view["put_call"]:
        r = "n/a" if p["ratio"] is None else f"{p['ratio']:.2f}"
        lines.append(f"  {p['expiry']}  calls {p['call_volume']:>9,.0f}  puts {p['put_volume']:>9,.0f}  ratio {r}")
    return "\n".join(lines)


def _n(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def status(store: ZeroStore, cfg: ZeroDteConfig) -> str:
    span = store.bar_span(cfg.symbol)
    lines = []
    if span:
        days = {B.et_day(ts) for ts, *_ in store.bars(cfg.symbol)}
        lines.append(f"1-minute bars: {span[2]:,} across {_n(len(days), 'day')}, "
                     f"{B.et_day(span[0])} to {B.et_day(span[1])}")
    else:
        lines.append("1-minute bars: none yet (run backfill)")
    snaps, oi = store.snapshot_days(cfg.symbol), store.oi_days(cfg.symbol)
    lines.append(f"Chain snapshots: {sum(snaps.values()):,} on {_n(len(snaps), 'day')}"
                 + (f" (last {max(snaps)})" if snaps else ""))
    lines.append(f"Open interest: {_n(len(oi), 'day')}" + (f" (last {oi[-1]})" if oi else ""))
    upcoming = cal.next_events(date.today(), 4)
    lines.append("Next news days: " + ", ".join(f"{d} {n}" for d, n in upcoming))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="0DTE QQQ data pipeline (free sources).")
    ap.add_argument("command", choices=["backfill", "snapshot", "watch", "levels", "status"])
    ap.add_argument("--since", default=None, help="backfill start date (default 2021-01-01)")
    ap.add_argument("--day", default=None, help="day for `levels` (default today)")
    args = ap.parse_args(argv)
    cfg = ZeroDteConfig()
    store = ZeroStore(cfg.db_path)
    try:
        if args.command == "status":
            print(status(store, cfg))
            return 0
        if args.command == "levels":
            day = date.fromisoformat(args.day) if args.day else datetime.now(B.ET).date()
            print(report(today_view(store, cfg, day), day))
            return 0
        from agent.config import ConfigError, load_settings
        try:
            src = AlpacaSources(load_settings())
        except ConfigError as err:
            print(f"Setup problem: {err}")
            return 1
        if args.command == "backfill":
            since = date.fromisoformat(args.since or cfg.backfill_since)
            n = B.backfill(store, src.stocks, cfg.symbol, since)
            print(f"Saved {n:,} bars.\n" + status(store, cfg))
        elif args.command == "snapshot":
            print(snapshot(store, src, cfg))
        else:
            return watch(store, src, cfg)
        return 0
    finally:
        store.close()


def watch(store: ZeroStore, src, cfg: ZeroDteConfig, sleep=_time.sleep, clock=None) -> int:
    """Snapshot every few minutes while the market is open; exit after the close."""
    clock = clock or (lambda: datetime.now(timezone.utc))
    seen_open = False
    while True:
        now = clock()
        et = now.astimezone(B.ET)
        try:
            is_open = src.market_open()
        except Exception as err:
            print(f"{et:%H:%M} clock check failed ({type(err).__name__}); retrying")
            is_open = None
        if is_open:
            seen_open = True
            res = snapshot(store, src, cfg, now)
            errs = [k for k in res if k.endswith("_error")]
            print(f"{et:%H:%M} QQQ {res['spot']:.2f}: {res.get('contracts', 0)} contracts "
                  f"({res.get('chain_source', '-')}), {res.get('bars', 0)} bars"
                  + (f", problems: {', '.join(errs)}" if errs else ""), flush=True)
        elif is_open is False and (seen_open or et.hour >= 16):
            print(f"{et:%H:%M} market closed, done for today")
            return 0
        elif is_open is False:
            print(f"{et:%H:%M} market not open yet, waiting", flush=True)
        sleep(cfg.snapshot_minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
