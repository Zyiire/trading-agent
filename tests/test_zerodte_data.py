"""Phase 2 of the 0DTE bot: the data pipeline, tested with made-up data (no network)."""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import pytest

from agent.stan.zerodte import bars as B
from agent.stan.zerodte import calendar as cal
from agent.stan.zerodte import chain as C
from agent.stan.zerodte import collect, flow, gex
from agent.stan.zerodte.config import ZeroDteConfig
from agent.stan.zerodte.levels import key_levels, merge, round_numbers
from agent.stan.zerodte.store import ZeroStore

DAY = date(2026, 11, 13)  # a Friday
OPEN_TS = B.day_bounds(DAY)[0]


def minute(i, o=600.0, c=None, v=100.0, h=None, l=None, base=OPEN_TS):
    c = o if c is None else c
    return (base + 60 * i, o, h if h is not None else max(o, c), l if l is not None else min(o, c), c, v, None)


# ----- bars -----
def test_three_minute_bars_group_session_minutes():
    rows = [minute(0, 600, 601, h=602), minute(1, 601, 599, l=598), minute(2, 599, 600), minute(3, 600, 603)]
    rows.append(minute(-5))          # pre-market: dropped
    rows.append(minute(390))         # 4:00 PM: dropped
    out = B.session_bars(rows)
    bars = out[DAY]
    assert [b["slot"] for b in bars] == [0, 1]
    b = bars[0]
    assert (b["o"], b["h"], b["l"], b["c"], b["v"]) == (600, 602, 598, 600, 300)
    assert b["ts"] == OPEN_TS and bars[1]["ts"] == OPEN_TS + 180


def test_unfinished_bar_is_left_out_live():
    rows = [minute(i) for i in range(5)]
    out = B.session_bars(rows, now_ts=OPEN_TS + 5 * 60)  # 9:35: only the 9:30 bar has closed
    assert [b["slot"] for b in out[DAY]] == [0]
    assert len(B.session_bars(rows, now_ts=OPEN_TS + 6 * 60)[DAY]) == 2


def test_session_vwap_is_cumulative_and_volume_weighted():
    rows = [minute(0, 100, 100, v=100), minute(1, 100, 100, v=100), minute(2, 100, 100, v=100),
            minute(3, 110, 110, v=300), minute(4, 110, 110, v=0), minute(5, 110, 110, v=0)]
    bars = B.session_bars(rows)[DAY]
    assert bars[0]["vwap"] == pytest.approx(100)
    assert bars[1]["vwap"] == pytest.approx((100 * 300 + 110 * 300) / 600)


def test_days_split_on_new_york_dates():
    nxt = B.day_bounds(DAY + timedelta(days=3))[0]
    out = B.session_bars([minute(0), minute(0, base=nxt)])
    assert sorted(out) == [DAY, DAY + timedelta(days=3)]


def test_day_bounds_handle_daylight_saving():
    summer, winter = B.day_bounds(date(2026, 7, 1))[0], B.day_bounds(date(2026, 12, 1))[0]
    assert datetime.fromtimestamp(summer, timezone.utc).hour == 13
    assert datetime.fromtimestamp(winter, timezone.utc).hour == 14


def test_yahoo_frame_rows():
    idx = pd.DatetimeIndex([pd.Timestamp(OPEN_TS, unit="s", tz="UTC")])
    df = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]}, index=idx)
    assert B.frame_rows(df) == [(OPEN_TS, 1.0, 2.0, 0.5, 1.5, 10.0, None)]


class FakeBarClient:
    def __init__(self):
        self.calls = []

    def get_stock_bars(self, req):
        self.calls.append((req.start, req.end, str(req.feed)))
        bar = type("Bar", (), {"timestamp": req.start.replace(tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1,
                                "volume": 5, "vwap": 1})
        return type("Set", (), {"data": {"QQQ": [bar]}})


def test_backfill_resumes_and_stops_16_minutes_ago():
    store, client = ZeroStore(":memory:"), FakeBarClient()
    now = datetime(2021, 3, 1, 15, 0, tzinfo=timezone.utc)
    B.backfill(store, client, "QQQ", date(2021, 1, 1), now=now, log=lambda *_: None)
    assert client.calls[-1][1].replace(tzinfo=None) == (now - timedelta(minutes=16)).replace(tzinfo=None)
    assert all("SIP" in c[2].upper() for c in client.calls)
    first, last_saved = len(client.calls), store.bar_span("QQQ")[1]
    B.backfill(store, client, "QQQ", date(2021, 1, 1), now=now, log=lambda *_: None)
    assert client.calls[first][0].replace(tzinfo=None) == datetime.fromtimestamp(last_saved + 60, timezone.utc).replace(tzinfo=None)


def test_live_bars_do_not_stop_the_full_market_backfill():
    store = ZeroStore(":memory:")
    store.save_bars("QQQ", [minute(0)], "alpaca-iex")
    assert store.bar_span("QQQ", source="alpaca-sip") is None
    assert store.bar_span("QQQ")[2] == 1


# ----- calendar -----
def test_calendar_dates():
    assert cal.events_on(date(2026, 11, 10)) == ["CPI"]
    assert cal.events_on(date(2026, 11, 6)) == ["jobs report"]
    assert cal.events_on(date(2026, 12, 9)) == ["FOMC"]
    assert cal.events_on(DAY) == []
    assert cal.is_news_day(date(2019, 6, 5)) is None  # outside the calendar
    assert cal.next_events(date(2026, 11, 17), 2) == [(date(2026, 12, 4), "jobs report"),
                                                       (date(2026, 12, 9), "FOMC")]


def test_calendar_has_one_cpi_and_one_jobs_report_most_months():
    for name in ("CPI", "jobs report"):
        months = [(d.year, d.month) for d in cal.EVENTS[name] if 2022 <= d.year <= 2024]
        assert len(months) == len(set(months)) == 36
    assert all(d.weekday() < 5 for days in cal.EVENTS.values() for d in days)
    assert len([d for d in cal.EVENTS["FOMC"] if d.year == 2024]) == 8


# ----- chain parsing -----
RAW = {"snapshots": {
    "QQQ261113C00600000": {"latestQuote": {"bp": 1.0, "ap": 1.2}, "latestTrade": {"p": 1.1},
                           "impliedVolatility": 0.2, "greeks": {"delta": 0.5, "gamma": 0.1},
                           "dailyBar": {"v": 1500}},
    "QQQ261113P00590000": {"latestQuote": {"bp": 0.4, "ap": 0.5}, "dailyBar": {"v": 3000}},
    "not-a-contract": {},
}}


def test_parse_alpaca_raw_chain():
    rows = sorted(C.parse_alpaca_raw(RAW), key=lambda r: r.strike)
    assert len(rows) == 2
    put, call = rows
    assert (call.expiry, call.strike, call.type, call.volume, call.iv) == (DAY, 600.0, "call", 1500, 0.2)
    assert call.mid == pytest.approx(1.1)
    assert put.type == "put" and put.iv is None and put.volume == 3000
    assert C.parse_alpaca_raw({"QQQ261113C00600000": {}})[0].strike == 600.0  # unwrapped shape too


def test_parse_yahoo_frame():
    df = pd.DataFrame([{"strike": 600.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1, "impliedVolatility": 0.25,
                        "volume": float("nan"), "openInterest": 900}])
    r = C.parse_yahoo_frame(df, DAY, "put")[0]
    assert (r.type, r.oi, r.iv, r.volume) == ("put", 900, 0.25, None)


class FakeContracts:
    def __init__(self):
        self.pages = 0

    def get_option_contracts(self, req):
        self.pages += 1
        mk = lambda k, t, oi: type("C", (), {"expiration_date": DAY, "strike_price": k,
                                             "type": type("E", (), {"value": t}), "open_interest": oi})
        if self.pages == 1:
            return type("R", (), {"option_contracts": [mk(600, "call", "1200"), mk(601, "call", None)],
                                  "next_page_token": "p2"})
        return type("R", (), {"option_contracts": [mk(590, "put", 800)], "next_page_token": None})


def test_alpaca_open_interest_pages_and_skips_missing():
    fake = FakeContracts()
    rows = C.alpaca_open_interest(fake, "QQQ", DAY, 7, 580, 620)
    assert fake.pages == 2
    assert rows == [(DAY, 600.0, "call", 1200.0), (DAY, 590.0, "put", 800.0)]


def test_open_interest_attached_from_store():
    rows = [C.ChainRow(DAY, 600.0, "call"), C.ChainRow(DAY, 590.0, "put", oi=5)]
    out = C.with_open_interest(rows, {(DAY.isoformat(), 600.0, "call"): 1200, (DAY.isoformat(), 590.0, "put"): 9})
    assert [r.oi for r in out] == [1200, 5]


# ----- put/call -----
def test_put_call_ratios():
    rows = [C.ChainRow(DAY, 600, "call", 1.0, 1.0, volume=1000), C.ChainRow(DAY, 590, "put", 0.5, 0.5, volume=3000),
            C.ChainRow(DAY + timedelta(days=3), 600, "call", 2, 2, volume=10)]
    pc = flow.put_call(rows, DAY)
    assert pc["ratio"] == pytest.approx(3.0)
    assert pc["premium_ratio"] == pytest.approx(3000 * 0.5 / (1000 * 1.0))
    assert [p["expiry"] for p in flow.by_expiry(rows)] == [DAY, DAY + timedelta(days=3)]
    assert flow.put_call([C.ChainRow(DAY, 1, "put", volume=5)])["ratio"] is None


# ----- GEX -----
NOON = OPEN_TS + int(2.5 * 3600)


def test_gamma_matches_a_finite_difference_of_delta():
    def delta(s, k=600, t=0.01, v=0.2):
        d1 = (math.log(s / k) + v * v / 2 * t) / (v * math.sqrt(t))
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    h = 0.01
    fd = (delta(600 + h) - delta(600 - h)) / (2 * h)
    assert gex.bs_gamma(600, 600, 0.01, 0.2) == pytest.approx(fd, rel=1e-4)


def test_years_to_expiry():
    assert gex.years_to_expiry(NOON, DAY) == pytest.approx(4 / (365 * 24))
    assert gex.years_to_expiry(B.day_bounds(DAY)[1] + 1, DAY) is None


def test_walls_signs_and_zero_gamma():
    wk = DAY + timedelta(days=7)  # a week out, so open interest (not closeness) decides the walls
    rows = [C.ChainRow(wk, 610.0, "call", iv=0.2, oi=50_000), C.ChainRow(wk, 605.0, "call", iv=0.2, oi=5_000),
            C.ChainRow(wk, 590.0, "put", iv=0.2, oi=60_000), C.ChainRow(wk, 595.0, "put", iv=0.2, oi=4_000)]
    s = gex.summary(rows, 600.0, NOON)
    assert s["call_wall"] == 610.0 and s["put_wall"] == 590.0
    cells = gex.by_strike(rows, 600.0, NOON)
    assert cells[610.0]["call"] > 0 and cells[590.0]["put"] < 0
    zg = s["zero_gamma"]
    assert zg is not None and 590 < zg < 610
    assert gex.total(rows, 608, NOON) > 0 > gex.total(rows, 592, NOON)


def test_same_day_gamma_crowds_next_to_the_price():
    # With hours left, a strike $5 away outweighs one $10 away holding 10x the open interest.
    rows = [C.ChainRow(DAY, 610.0, "call", iv=0.2, oi=50_000), C.ChainRow(DAY, 605.0, "call", iv=0.2, oi=5_000)]
    assert gex.summary(rows, 600.0, NOON)["call_wall"] == 605.0


def test_gex_skips_rows_without_oi_or_iv_and_uses_the_fallback():
    rows = [C.ChainRow(DAY, 600.0, "call", oi=100), C.ChainRow(DAY, 600.0, "put", iv=0.2)]
    assert gex.by_strike(rows, 600, NOON) == {}
    assert gex.by_strike(rows, 600, NOON, iv_fallback=0.2)[600.0]["call"] > 0
    assert gex.summary([], 600, NOON)["net"] is None


# ----- levels -----
def test_round_numbers():
    cfg = ZeroDteConfig(level_window=0.01)
    got = round_numbers(681.0, cfg)
    assert (680.0, "round10") in got and (685.0, "round5") in got and (690.0, "round10") not in got


def test_merge_keeps_every_reason():
    out = merge([(680.0, "round10"), (680.3, "call_wall"), (690.0, "round10")], 685.0, 0.001)
    assert out[0]["price"] == 680.0 and out[0]["kinds"] == ["call_wall", "round10"]
    assert out[0]["strength"] == 5 and out[0]["side"] == "below" and out[1]["side"] == "above"


def test_key_levels_include_prior_day_open_and_walls():
    prior = [{"o": 595, "h": 603.5, "l": 594.2, "c": 601.1}]
    today = [{"o": 601.7, "h": 602, "l": 601, "c": 601.5}]
    rows = [C.ChainRow(DAY, 610.0, "call", iv=0.2, oi=50_000), C.ChainRow(DAY, 590.0, "put", iv=0.2, oi=60_000)]
    view = key_levels(600.0, prior, today, rows, NOON)
    kinds = {k for lv in view["levels"] for k in lv["kinds"]}
    assert {"prior_high", "prior_low", "prior_close", "open", "call_wall", "put_wall", "round10",
            "call_oi", "put_oi"} <= kinds
    assert next(lv for lv in view["levels"] if lv["price"] == 610.0)["kinds"][0] == "call_wall"


# ----- store + collector -----
class FakeSources:
    def __init__(self, fail_chain=False):
        self.fail_chain, self.oi_calls, self.open = fail_chain, 0, [True, True, False]

    def market_open(self):
        return self.open.pop(0)

    def spot(self, symbol):
        return 600.0, "test"

    def today_bars(self, symbol, day):
        start = B.day_bounds(day)[0]
        return [minute(i, base=start) for i in range(6)], "test"

    def chain(self, symbol, today, max_dte, lo, hi):
        if self.fail_chain:
            raise TimeoutError("chain down")
        return [C.ChainRow(today, 610.0, "call", 1, 1.2, iv=0.2, volume=1000),
                C.ChainRow(today, 590.0, "put", 0.8, 1.0, iv=0.2, volume=2500)], "test"

    def open_interest(self, symbol, today, max_dte, lo, hi):
        self.oi_calls += 1
        return [(today, 610.0, "call", 50_000), (today, 590.0, "put", 60_000)], "test"


def test_snapshot_saves_everything_and_open_interest_once_a_day():
    store, src = ZeroStore(":memory:"), FakeSources()
    now = datetime.fromtimestamp(NOON, timezone.utc)
    res = collect.snapshot(store, src, ZeroDteConfig(), now)
    assert (res["contracts"], res["open_interest"], res["bars"]) == (2, 2, 6)
    collect.snapshot(store, src, ZeroDteConfig(), now + timedelta(minutes=5))
    assert src.oi_calls == 1
    assert len(store.snapshot_times("QQQ")) == 2


def test_snapshot_reports_a_failed_source_and_keeps_the_rest():
    store = ZeroStore(":memory:")
    res = collect.snapshot(store, FakeSources(fail_chain=True), ZeroDteConfig(), datetime.fromtimestamp(NOON, timezone.utc))
    assert "chain_error" in res and res["bars"] == 6 and res["open_interest"] == 2


def test_levels_view_and_report():
    store, cfg = ZeroStore(":memory:"), ZeroDteConfig()
    collect.snapshot(store, FakeSources(), cfg, datetime.fromtimestamp(NOON, timezone.utc))
    view = collect.today_view(store, cfg, DAY)
    assert view["gex"]["call_wall"] == 610.0 and view["put_call"][0]["ratio"] == pytest.approx(2.5)
    text = collect.report(view, DAY)
    assert "call wall 610.0" in text and "ratio 2.50" in text and "News day: none" in text
    assert "No snapshot" in collect.report(collect.today_view(store, cfg, DAY + timedelta(days=3)), DAY)
    assert "Chain snapshots: 1 on 1 day" in collect.status(store, cfg)


def test_watch_runs_until_the_close():
    store, src, cfg = ZeroStore(":memory:"), FakeSources(), ZeroDteConfig()
    ticks = iter(datetime.fromtimestamp(NOON + 300 * i, timezone.utc) for i in range(10))
    naps = []
    assert collect.watch(store, src, cfg, sleep=naps.append, clock=lambda: next(ticks)) == 0
    assert len(store.snapshot_times("QQQ")) == 2 and naps == [300, 300]


def test_store_upserts_and_drops_nan():
    store = ZeroStore(":memory:")
    store.save_bars("QQQ", [minute(0, v=float("nan"))], "x")
    store.save_bars("QQQ", [minute(0, 601)], "x")
    rows = store.bars("QQQ")
    assert len(rows) == 1 and rows[0][1] == 601
    store.save_bars("QQQ", [minute(1, v=float("nan"))], "x")
    assert store.bars("QQQ")[1][5] is None
