"""Offline tests. They use fake Alpaca clients, so no keys or network are needed."""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from agent.broker.alpaca import AlpacaBroker
from agent.broker.base import DailyBar, OptionQuote, parse_occ_symbol
from agent.config import ConfigError, Settings, load_settings
from agent.jobs.backfill_bars import HISTORY_START, backfill
from agent.jobs.snapshot_chains import snapshot_once
from agent.store import Store

SETTINGS = Settings(api_key="k", secret_key="s")


# ---------- config ----------
def test_missing_keys_is_an_error(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_live_mode_is_locked(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    with pytest.raises(ConfigError, match="locked"):
        load_settings(env_file=None)


def test_defaults_match_the_plan(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    monkeypatch.setenv("WATCHLIST", "spy, aapl")
    s = load_settings(env_file=None)
    assert s.paper is True
    assert s.starting_capital == 50.0
    assert s.kill_switch_equity == 40.0
    assert s.watchlist == ("SPY", "AAPL")


def test_keys_never_show_in_repr():
    assert "secret-value" not in repr(Settings(api_key="key-value", secret_key="secret-value"))
    assert "key-value" not in repr(Settings(api_key="key-value", secret_key="secret-value"))


def test_broker_refuses_live_settings():
    with pytest.raises(RuntimeError):
        AlpacaBroker(Settings(api_key="k", secret_key="s", paper=False), object(), object(), object())


# ---------- OCC symbols and quote maths ----------
def test_parse_occ_symbol():
    assert parse_occ_symbol("AAPL261218C00150000") == ("AAPL", date(2026, 12, 18), "call", 150.0)
    assert parse_occ_symbol("SPY270115P00612500") == ("SPY", date(2027, 1, 15), "put", 612.5)
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL")


def _quote(bid, ask, contract="AAPL261218C00150000"):
    return OptionQuote(contract, "AAPL", date(2026, 12, 18), "call", 150.0, bid, ask,
                       10, 12, 1.0, 0.3, 0.5, 0.02, -0.04, 0.2)


def test_mid_and_spread():
    q = _quote(1.00, 1.10)
    assert q.mid == 1.05
    assert q.spread_pct == pytest.approx(0.0952, abs=1e-4)
    assert _quote(None, 1.10).mid is None
    assert _quote(0.0, 0.0).spread_pct is None


# ---------- Alpaca adapter with fake clients ----------
class FakeTrading:
    def get_account(self):
        return SimpleNamespace(
            id="abc", status=SimpleNamespace(value="ACTIVE"), equity="50.00", cash="50.00",
            buying_power="50.00", options_buying_power="50.00", options_trading_level=2,
            pattern_day_trader=False, trading_blocked=False, multiplier="1",
        )

    def get_all_positions(self):
        return [SimpleNamespace(symbol="SPY", asset_class=SimpleNamespace(value="us_equity"),
                                qty="1", avg_entry_price="600.5", market_value="601",
                                unrealized_pl="0.5")]

    def get_clock(self):
        t = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)
        return SimpleNamespace(timestamp=t, is_open=True, next_open=t, next_close=t)


class FakeStocks:
    def __init__(self):
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        bar = SimpleNamespace(timestamp=datetime(2026, 11, 13, 5, 0, tzinfo=timezone.utc),
                              open=1, high=2, low=0.5, close=1.5, volume=100, vwap=1.2)
        return SimpleNamespace(data={"SPY": [bar]})


class FakeOptions:
    def __init__(self):
        self.requests = []

    def get_option_chain(self, request):
        self.requests.append(request)
        snap = SimpleNamespace(
            latest_quote=SimpleNamespace(bid_price=1.0, ask_price=1.1, bid_size=5, ask_size=7),
            latest_trade=SimpleNamespace(price=1.05), implied_volatility=0.25,
            greeks=SimpleNamespace(delta=0.4, gamma=0.01, theta=-0.03, vega=0.1),
        )
        empty = SimpleNamespace(latest_quote=None, latest_trade=None, implied_volatility=None, greeks=None)
        return {"SPY261218C00600000": snap, "SPY261218P00600000": empty, "garbage": snap}


def _broker():
    return AlpacaBroker(SETTINGS, FakeTrading(), FakeStocks(), FakeOptions())


def test_account_mapping():
    a = _broker().get_account()
    assert (a.status, a.equity, a.options_level, a.multiplier, a.is_paper) == ("ACTIVE", 50.0, 2, "1", True)


def test_positions_and_clock():
    b = _broker()
    assert b.get_positions()[0].symbol == "SPY"
    assert b.get_clock().is_open is True


def test_daily_bars_mapping():
    bars = _broker().get_daily_bars(["SPY"], start=date(2026, 11, 1))
    assert bars == [DailyBar("SPY", date(2026, 11, 13), 1.0, 2.0, 0.5, 1.5, 100.0, 1.2)]


def test_option_chain_mapping_skips_bad_symbols_and_handles_missing_fields():
    quotes = _broker().get_option_chain("SPY", date(2026, 11, 16), date(2027, 1, 15), 510, 690)
    assert len(quotes) == 2
    call = next(q for q in quotes if q.option_type == "call")
    put = next(q for q in quotes if q.option_type == "put")
    assert (call.strike, call.bid, call.delta, call.mid) == (600.0, 1.0, 0.4, 1.05)
    assert put.bid is None and put.delta is None and put.mid is None


# ---------- store and jobs ----------
class FakeBroker:
    def __init__(self, fail_on=()):
        self.bar_calls, self.chain_calls, self.fail_on = [], [], set(fail_on)

    def get_daily_bars(self, symbols, start, end=None):
        self.bar_calls.append((tuple(symbols), start))
        return [DailyBar(symbols[0], date(2026, 11, 12), 1, 2, 0.5, 100.0, 10, None),
                DailyBar(symbols[0], date(2026, 11, 13), 1, 2, 0.5, 200.0, 10, None)]

    def get_option_chain(self, underlying, expiration_from, expiration_to, strike_min=None, strike_max=None):
        self.chain_calls.append((underlying, expiration_from, expiration_to, strike_min, strike_max))
        if underlying in self.fail_on:
            raise RuntimeError("boom")
        return [_quote(1.0, 1.1, f"{underlying}261218C00150000")]


def test_backfill_is_incremental():
    store, broker = Store(":memory:"), FakeBroker()
    assert backfill(broker, store, ["SPY"], today=date(2026, 11, 14)) == {"SPY": 2}
    assert broker.bar_calls[0][1] == HISTORY_START
    assert store.last_bar_day("SPY") == date(2026, 11, 13)
    assert store.last_close("SPY") == 200.0
    backfill(broker, store, ["SPY"], today=date(2026, 11, 14))
    assert broker.bar_calls[1][1] == date(2026, 11, 14)  # only asks for new days
    assert store.count_bars("SPY") == 2  # no duplicates
    assert backfill(broker, store, ["SPY"], today=date(2026, 11, 13)) == {"SPY": 0}


def test_snapshot_uses_strike_band_and_survives_one_failure():
    store, broker = Store(":memory:"), FakeBroker(fail_on={"BAD"})
    store.save_daily_bars([DailyBar("SPY", date(2026, 11, 13), 1, 2, 0.5, 200.0, 10, None)])
    saved = snapshot_once(broker, store, ["SPY", "BAD", "NEW"], SETTINGS,
                          today=date(2026, 11, 16), taken_at="2026-11-16T15:00:00+00:00")
    assert saved == {"SPY": 1, "BAD": -1, "NEW": 1}
    spy_call = broker.chain_calls[0]
    assert spy_call[1:3] == (date(2026, 11, 16), date(2027, 1, 15))
    assert spy_call[3:] == (170.0, 230.0)  # 15% either side of 200
    assert broker.chain_calls[2][3:] == (None, None)  # no price history yet
    assert store.count_option_rows() == 2
    # same timestamp again must not duplicate rows
    snapshot_once(broker, store, ["SPY"], SETTINGS, today=date(2026, 11, 16),
                  taken_at="2026-11-16T15:00:00+00:00")
    assert store.count_option_rows() == 2


def test_account_snapshot_saves(tmp_path):
    store = Store(tmp_path / "nested" / "agent.db")
    store.save_account_snapshot(_broker().get_account(), "2026-11-16T15:00:00+00:00")
    store.close()
    assert (tmp_path / "nested" / "agent.db").exists()
