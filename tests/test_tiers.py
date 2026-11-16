"""The three arena winners on the paper account: plans, orders, switching, and the kill switch."""
from datetime import date, datetime, time, timezone

import numpy as np
import pandas as pd
import pytest

from agent.stan import tiers
from agent.stan.paper import PaperBook, run_evening, snapshot
from agent.stan.shadow import ET
from tests.test_paper import FakeAlpaca


def _frame(closes, days):
    c = pd.Series(closes, index=days)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.005, "low": c * 0.995,
                         "close": c, "volume": 1e6})


def _market(n=300, qqq_trend=0.001, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-06-02", periods=n)
    spy = 500 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    qqq = 400 * np.cumprod(1 + qqq_trend + rng.normal(0, 0.012, n))
    tqqq = 50 * np.cumprod(1 + 3 * np.diff(np.concatenate([[qqq[0]], qqq])) / np.concatenate([[qqq[0]], qqq[:-1]]))
    return {"SPY": _frame(spy, days), "QQQ": _frame(qqq, days), "TQQQ": _frame(tqqq, days)}


# ---------------------------------------------------------------- plans
def test_month_end_detection():
    assert tiers.is_month_end(date(2026, 10, 30))       # Friday Oct 30 is the last business day
    assert not tiers.is_month_end(date(2026, 10, 29))


def test_low_and_medium_size_by_last_month_end_volatility():
    h = _market()
    med, low = tiers.plan("medium", h), tiers.plan("low", h, low_fraction=0.5)
    assert 0 < med.weights["SPY"] <= 1 and med.weights["TQQQ"] == 0
    assert low.weights["SPY"] == pytest.approx(med.weights["SPY"] * 0.5, abs=1e-3)
    assert date.fromisoformat(med.signal_day) <= h["SPY"].index[-1].date()


def test_sizing_ignores_days_after_the_month_end():
    """Mid-month moves must not change the size: it's set at the month end."""
    h = _market()
    days = h["SPY"].index
    k = max(i for i in range(len(days) - 1) if days[i].month != days[i + 1].month)   # last month end
    cut = {s: df.iloc[:k + 6] for s, df in h.items()}                                 # 5 days into the new month
    wild = {s: df.copy() for s, df in cut.items()}
    wild["SPY"].iloc[-3:, wild["SPY"].columns.get_loc("close")] *= [1.08, 0.9, 1.1]  # violent days after it
    assert tiers.plan("medium", cut).weights == tiers.plan("medium", wild).weights


def test_high_holds_tqqq_only_above_the_200_day_average():
    up, down = _market(qqq_trend=0.002), _market(qqq_trend=-0.002)
    assert tiers.plan("high", up).weights == {"SPY": 0.0, "TQQQ": 1.0}
    assert tiers.plan("high", down).weights == {"SPY": 0.0, "TQQQ": 0.0}


# ---------------------------------------------------------------- orders
def test_orders_respect_the_band_and_sell_first():
    p = tiers.TierPlan("high", {"SPY": 0.0, "TQQQ": 1.0}, "", "2026-01-01")
    todo = tiers.orders_for(p, 10_000, {"SPY": 5.0}, {"SPY": 500.0, "TQQQ": 50.0})
    assert todo[0] == ("SPY", "sell", 5.0) and todo[1][0:2] == ("TQQQ", "buy")
    near = tiers.TierPlan("medium", {"SPY": 0.62, "TQQQ": 0.0}, "", "2026-01-01")
    assert tiers.orders_for(near, 10_000, {"SPY": 12.0}, {"SPY": 500.0}) == []      # 60% held, 62% target


# ---------------------------------------------------------------- the evening run
def _walk(broker, book, h, start, stop):
    days = h["SPY"].index
    out = []
    for k in range(start, stop):
        cut = {s: df[df.index <= days[k]] for s, df in h.items()}
        broker.mark({s: float(df["close"].iloc[-1]) for s, df in cut.items()})
        out.append(run_evening(broker, book, cut, now=datetime.combine(days[k].date(), time(17, 0), tzinfo=ET)))
        if k + 1 < len(days):
            nxt = days[k + 1]
            broker.open_auction({s: float(df["open"].loc[nxt]) for s, df in h.items()},
                                datetime.combine(nxt.date(), time(14, 30), tzinfo=timezone.utc))
    return out


def test_medium_tier_builds_and_keeps_the_spy_position():
    h, broker, book = _market(), FakeAlpaca(cash=10_000), PaperBook(":memory:")
    book.set("strategy", "medium")
    _walk(broker, book, h, 250, 270)
    px = float(h["SPY"]["close"].iloc[269])
    w = broker.positions.get("SPY", 0) * px / broker.equity()
    target = tiers.plan("medium", {s: df.iloc[:270] for s, df in h.items()}).weights["SPY"]
    assert abs(w - target) <= tiers.BAND + 0.02
    assert snapshot(book)["strategy"] == "medium" and snapshot(book)["tier_plan"]["tier"] == "medium"


def test_one_set_of_orders_per_day():
    h, broker, book = _market(qqq_trend=0.002), FakeAlpaca(cash=10_000), PaperBook(":memory:")
    book.set("strategy", "high")
    days = h["SPY"].index
    cut = {s: df[df.index <= days[260]] for s, df in h.items()}
    broker.mark({s: float(df["close"].iloc[-1]) for s, df in cut.items()})
    now = datetime.combine(days[260].date(), time(17, 0), tzinfo=ET)
    run_evening(broker, book, cut, now=now)
    run_evening(broker, book, cut, now=now)   # a second run the same evening sends nothing new
    assert sum(1 for o in broker.orders.values() if o["symbol"] == "TQQQ") == 1


def test_dry_run_sends_nothing():
    h, broker, book = _market(qqq_trend=0.002), FakeAlpaca(cash=10_000), PaperBook(":memory:")
    book.set("strategy", "high")
    days = h["SPY"].index
    cut = {s: df[df.index <= days[260]] for s, df in h.items()}
    broker.mark({s: float(df["close"].iloc[-1]) for s, df in cut.items()})
    res = run_evening(broker, book, cut, now=datetime.combine(days[260].date(), time(17, 0), tzinfo=ET),
                      dry_run=True)
    assert not broker.orders and any("Would buy" in a for a in res["actions"])


def test_switching_waits_for_open_selloff_trades():
    h, broker, book = _market(), FakeAlpaca(cash=10_000), PaperBook(":memory:")
    book.add_trade({"symbol": "AAA", "signal_date": "2026-01-02", "status": "open", "qty": 3, "entry_order": "x"})
    broker.positions["AAA"] = 3
    h["AAA"] = h["SPY"].copy()
    book.set("strategy", "medium")
    days = h["SPY"].index
    cut = {s: df[df.index <= days[260]] for s, df in h.items()}
    broker.mark({s: float(df["close"].iloc[-1]) for s, df in cut.items()})
    res = run_evening(broker, book, cut, now=datetime.combine(days[260].date(), time(17, 0), tzinfo=ET))
    assert any("waiting for 1 selloff trade" in a for a in res["actions"])
    assert not any(o["symbol"] == "SPY" for o in broker.orders.values())


def test_kill_switch_sells_tier_holdings_and_stops():
    h, broker, book = _market(qqq_trend=0.002), FakeAlpaca(cash=10_000), PaperBook(":memory:")
    book.set("strategy", "high")
    _walk(broker, book, h, 255, 258)
    assert broker.positions.get("TQQQ", 0) > 0
    broker.marks["TQQQ"] = broker.marks["TQQQ"] * 0.3        # a crash takes the account under the floor
    days = h["SPY"].index
    cut = {s: df[df.index <= days[258]] for s, df in h.items()}
    res = run_evening(broker, book, cut, now=datetime.combine(days[258].date(), time(17, 0), tzinfo=ET))
    assert res["halted"]
    assert any(o["symbol"] == "TQQQ" and o["side"] == "sell" for o in broker.orders.values())
