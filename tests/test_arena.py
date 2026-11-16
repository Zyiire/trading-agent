"""Strategy arena: option pricing, option books, no lookahead in signals, tiers and ranking."""
import math

import numpy as np
import pandas as pd
import pytest

from agent.stan import arena as ar
from agent.stan.train import DEFAULT_SYMBOLS, demo_history

CFG = ar.ArenaConfig()


@pytest.fixture(scope="module")
def hist():
    h = demo_history(DEFAULT_SYMBOLS[:10], years=6, seed=8)
    h.update(ar.demo_extras(h))
    return h


# ---------------------------------------------------------------- pricing
def test_black_scholes_known_value_and_parity():
    c = ar.bs("call", 100, 100, 1.0, 0.2, 0.05)
    p = ar.bs("put", 100, 100, 1.0, 0.2, 0.05)
    assert c == pytest.approx(10.4506, abs=1e-3)               # textbook value
    assert c - p == pytest.approx(100 - 100 * math.exp(-0.05), abs=1e-9)   # put-call parity
    assert ar.bs("put", 90, 100, 0.0, 0.2) == 10               # expiry: intrinsic value


def _flat_data(n=80, vol=0.20, price=100.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame({"SPY": price, "QQQ": price}, index=idx, dtype=float)
    v = pd.Series(vol / CFG.iv_scale, index=idx)
    return ar.Data(close, pd.Series(0.0, index=idx), v, v, {})


def test_bought_options_lose_their_premium_when_the_price_never_moves():
    d = _flat_data()
    r = ar.option_book(d, CFG, lambda t: [("SPY", "call", 0.05)], "buy")
    first = (1 + r.iloc[:CFG.opt_days + 1]).prod() - 1
    assert first == pytest.approx(-0.05, abs=1e-9)             # the whole 5% budget expires worthless


def test_sold_puts_keep_the_premium_when_the_price_never_moves():
    d = _flat_data()
    r = ar.option_book(d, CFG, lambda t: [("SPY", "put", 1.0)], "putwrite")
    first = (1 + r.iloc[:CFG.opt_days + 1]).prod() - 1
    prem = ar.bs("put", 100, 100, CFG.opt_days / 252, 0.20)
    assert first == pytest.approx(prem * (1 - CFG.opt_cost) / 100, rel=1e-6)


def test_bought_put_pays_off_in_a_crash():
    d = _flat_data()
    d.close.loc[d.close.index[10]:, "QQQ"] = 80.0                 # a 20% drop inside the first month
    r = ar.option_book(d, CFG, lambda t: [("QQQ", "put", 0.05)], "buy")
    prem = ar.bs("put", 100, 100, CFG.opt_days / 252, 0.20)
    expected = 0.05 * 20 / (prem * (1 + CFG.opt_cost)) - 0.05     # $20 of intrinsic value per put at expiry
    assert (1 + r.iloc[:CFG.opt_days + 1]).prod() - 1 == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------- no lookahead
def test_signals_only_use_the_past(hist):
    d = ar.make_data(hist)
    full = ar.above_avg(d.close["SPY"], 200)
    for t in range(300, len(d.close), 97):
        cut = ar.above_avg(d.close["SPY"].iloc[:t + 1], 200)
        assert cut.iloc[-1] == full.iloc[t]


def test_monthly_strategies_dont_change_when_the_future_is_cut_off(hist):
    d = ar.make_data(hist)
    full = ar.s_gem(d, CFG, 1.0)
    for t in range(400, len(d.close), 150):
        h_cut = {k: v[v.index <= d.close.index[t]] for k, v in hist.items()}
        cut = ar.s_gem(ar.make_data(h_cut), CFG, 1.0)
        assert cut.iloc[-1] == pytest.approx(full.iloc[t])


# ---------------------------------------------------------------- tiers and the arena
def test_tiers():
    assert ar.tier_of(-0.15) == "low" and ar.tier_of(-0.30) == "medium" and ar.tier_of(-0.55) == "high"


def test_arena_end_to_end(hist):
    out = ar.run_arena(hist)
    ran = [s for s in out["strategies"] if "skipped" not in s]
    assert len(ran) >= 16
    assert [s["rank"] for s in ran] == list(range(1, len(ran) + 1))
    for tier, key in out["winners"].items():
        if key:
            w = next(s for s in ran if s["key"] == key)
            assert w["eligible"] and w["tier"] == tier and not w["survivorship"]
    assert all(not s["eligible"] for s in ran if s["survivorship"])


def test_missing_data_is_skipped_not_crashed():
    h = demo_history(["SPY", "AAPL", "MSFT"], years=4, seed=3)
    out = ar.run_arena(h)
    skipped = {s["key"] for s in out["strategies"] if "skipped" in s}
    assert {"gem", "sectors", "buy_puts"} <= skipped


def test_stock_strategies_never_trade_etfs(hist):
    d = ar.make_data(hist)
    w_cols = [s for s in d.close.columns if s not in ar.NON_STOCKS]
    assert not set(w_cols) & set(ar.SECTORS) and "EFA" not in w_cols and "SPY" not in w_cols
