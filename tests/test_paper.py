"""Stan's paper trading, run day by day against a fake Alpaca paper account."""
from __future__ import annotations

from datetime import datetime, time, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent.broker.base import AccountInfo, OrderInfo, PositionInfo
from agent.stan.paper import PaperBook, PaperConfig, exit_due, order_id, run_evening, signals_today, snapshot
from agent.stan.shadow import ET
from agent.stan.strategies import BASE_RULE


class FakeAlpaca:
    """Market-on-open orders fill at the next open when `open_auction` runs."""

    def __init__(self, cash=100_000.0, paper=True):
        self.cash, self.paper = cash, paper
        self.positions: dict[str, float] = {}
        self.orders: dict[str, dict] = {}
        self.marks: dict[str, float] = {}
        self.partial: dict[str, float] = {}   # symbol -> share of the order that fills
        self.reject: set[str] = set()

    # account
    def equity(self):
        return self.cash + sum(q * self.marks.get(s, 0) for s, q in self.positions.items())

    def get_account(self):
        e = self.equity()
        return AccountInfo("acct", "ACTIVE", self.paper, e, self.cash, self.cash, None, 2, False, False, "1")

    def get_positions(self):
        return [PositionInfo(s, "us_equity", q, 0, None, None) for s, q in self.positions.items() if q]

    # orders
    def submit_open_order(self, symbol, qty, side, client_order_id):
        if client_order_id in self.orders:
            raise RuntimeError("client_order_id must be unique")
        if symbol in self.reject:
            raise RuntimeError("rejected by fake broker")
        self.orders[client_order_id] = {"symbol": symbol, "qty": qty, "side": side, "status": "accepted",
                                        "filled_qty": 0.0, "price": None, "at": None}
        return self.get_order(client_order_id)

    def submit_fractional_order(self, symbol, qty, side, client_order_id):
        out = self.submit_open_order(symbol, qty, side, client_order_id)
        self.orders[client_order_id]["frac"] = True
        return out

    def get_order(self, cid):
        o = self.orders.get(cid)
        if o is None:
            return None
        return OrderInfo(cid, o["symbol"], o["side"], o["qty"], o["status"], o["filled_qty"], o["price"], o["at"])

    def cancel_open_orders(self):
        n = 0
        for o in self.orders.values():
            if o["status"] == "accepted":
                o["status"] = "canceled"
                n += 1
        return n

    def open_auction(self, opens: dict[str, float], when: datetime):
        for o in self.orders.values():
            if o["status"] != "accepted":
                continue
            px = opens[o["symbol"]]
            share = self.partial.get(o["symbol"], 1.0)
            q = o["qty"] * share if o.get("frac") else int(o["qty"] * share)
            sign = 1 if o["side"] == "buy" else -1
            self.positions[o["symbol"]] = self.positions.get(o["symbol"], 0) + sign * q
            self.cash -= sign * q * px
            o.update(filled_qty=float(q), price=px, at=when, status="filled" if q == o["qty"] else "canceled")

    def mark(self, closes: dict[str, float]):
        self.marks.update(closes)


def _history():
    """AAA rises, sells off hard (RSI under 30), then rebounds. SPY and BBB drift up."""
    days = pd.bdate_range("2026-06-01", periods=90)
    a = 100 * 1.004 ** np.arange(50)
    fall = a[-1] * 0.975 ** np.arange(1, 13)
    rise = fall[-1] * 1.03 ** np.arange(1, 29)
    aaa = np.concatenate([a, fall, rise])
    out = {}
    for sym, closes in {"SPY": 400 * 1.001 ** np.arange(90), "BBB": 50 * 1.002 ** np.arange(90), "AAA": aaa}.items():
        c = pd.Series(closes, index=days)
        out[sym] = pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]) * 1.001, "high": c * 1.01,
                                 "low": c * 0.99, "close": c, "volume": 1e6})
    return out


def _evening(day):
    return datetime.combine(day.date(), time(17, 0), tzinfo=ET)


def _walk(broker, book, history, start, stop, cfg=PaperConfig()):
    days = history["SPY"].index
    results = []
    for k in range(start, stop):
        cut = {s: df[df.index <= days[k]] for s, df in history.items()}
        broker.mark({s: float(df["close"].iloc[-1]) for s, df in cut.items()})
        results.append(run_evening(broker, book, cut, now=_evening(days[k]), cfg=cfg))
        if k + 1 < len(days):
            nxt = days[k + 1]
            broker.open_auction({s: float(df["open"].loc[nxt]) for s, df in history.items()},
                                datetime.combine(nxt.date(), time(14, 30), tzinfo=timezone.utc))
    return results


def test_full_trade_round_trip():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    _walk(broker, book, history, 45, 85)
    trades = book.trades()
    aaa = [t for t in trades if t["symbol"] == "AAA"]
    assert aaa, "the selloff should have triggered a buy"
    first = aaa[-1]  # oldest
    assert first["status"] == "closed"
    assert first["entry_price"] and first["exit_price"] and first["entry_date"] < first["exit_date"]
    assert first["qty"] == int(100_000 * 0.10 // history["AAA"]["close"].loc[first["signal_date"]])
    assert first["ret"] == pytest.approx(first["exit_price"] / first["entry_price"] - 1, abs=1e-5)
    assert broker.positions.get("AAA", 0) == sum(t["qty"] for t in aaa if t["status"] in ("open", "pending_exit"))
    assert not any(t["symbol"] in ("SPY", "BBB") for t in trades)  # no selloff there, and SPY is the yardstick
    snap = snapshot(book)
    assert snap["enabled"] and snap["start_equity"] == pytest.approx(100_000)
    assert snap["stats"]["closed"] >= 1


def test_same_evening_twice_sends_nothing_new():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    days = history["SPY"].index
    for k in range(45, 70):  # find the first signal evening
        cut = {s: df[df.index <= days[k]] for s, df in history.items()}
        if signals_today(cut, days[k].date(), BASE_RULE):
            break
    broker.mark({s: float(df["close"].iloc[-1]) for s, df in cut.items()})
    run_evening(broker, book, cut, now=_evening(days[k]))
    n = len(broker.orders)
    run_evening(broker, book, cut, now=_evening(days[k]))
    assert n == 1 and len(broker.orders) == 1


def test_kill_switch_sells_everything_and_stops():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    _walk(broker, book, history, 45, 56)  # in the trade
    assert book.trades(["open"])
    broker.cash -= 30_000  # a disaster: equity falls more than 20%
    days = history["SPY"].index
    cut = {s: df[df.index <= days[56]] for s, df in history.items()}
    res = run_evening(broker, book, cut, now=_evening(days[56]))
    assert res["halted"] and "Kill switch" in book.get("halt_reason")
    assert not book.trades(["open"]) and book.trades(["pending_exit"])
    assert all(t["exit_reason"] == "kill switch" for t in book.trades(["pending_exit"]))
    # Halted: later evenings send no new buys.
    before = set(broker.orders)
    _walk(broker, book, history, 57, 60)
    assert not [cid for cid in broker.orders if cid not in before and cid.endswith("-buy")]


def test_mismatch_with_alpaca_halts():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    _walk(broker, book, history, 45, 56)
    broker.positions["AAA"] += 5  # someone traded AAA by hand
    days = history["SPY"].index
    cut = {s: df[df.index <= days[56]] for s, df in history.items()}
    res = run_evening(broker, book, cut, now=_evening(days[56]))
    assert res["halted"] and "AAA" in book.get("halt_reason")


def test_partial_fill_keeps_what_was_bought():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    broker.partial["AAA"] = 0.5
    _walk(broker, book, history, 45, 56)
    t = [t for t in book.trades() if t["symbol"] == "AAA"][-1]
    assert t["status"] in ("open", "pending_exit") and broker.positions["AAA"] == t["qty"]
    assert not book.get("halted", False)


def test_dry_run_sends_nothing():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    days = history["SPY"].index
    for k in range(45, 70):
        cut = {s: df[df.index <= days[k]] for s, df in history.items()}
        res = run_evening(broker, book, cut, now=_evening(days[k]), dry_run=True)
        if any("Buying" in a for a in res["actions"]):
            break
    assert any("Buying" in a for a in res["actions"])
    assert broker.orders == {} and book.trades() == [] and book.get("start_equity") is None


def test_refuses_a_live_account():
    history, book = _history(), PaperBook(":memory:")
    days = history["SPY"].index
    with pytest.raises(RuntimeError):
        run_evening(FakeAlpaca(paper=False), book, history, now=_evening(days[-1]))


def test_skips_stocks_held_outside_stans_trades():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    broker.positions["AAA"] = 10  # Zy already owns some AAA
    _walk(broker, book, history, 45, 70)
    assert not [t for t in book.trades() if t["symbol"] == "AAA"]
    assert any("outside my trades" in l["text"] for l in book.logs(500))


def test_broker_error_is_logged_not_fatal():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    broker.reject.add("AAA")
    _walk(broker, book, history, 45, 60)
    assert not book.trades()
    assert any("Couldn't send the AAA buy order" in l["text"] for l in book.logs(500))


def test_exit_rule_matches_the_backtest():
    days = pd.bdate_range("2026-01-01", periods=40)
    falling = pd.Series(100 * 0.98 ** np.arange(40), index=days)
    df = pd.DataFrame({"close": falling})
    sig = days[20].date().isoformat()
    assert exit_due(df.iloc[:25], sig, days[24].date(), BASE_RULE) is None  # 4 days, RSI still low
    assert "10 trading days" in exit_due(df.iloc[:31], sig, days[30].date(), BASE_RULE)


def test_order_ids_are_stable_and_unique():
    assert order_id("AAA", "2026-11-13", "buy") == "stan-AAA-20261113-buy"
    assert order_id("AAA", "2026-11-13", "buy") != order_id("AAA", "2026-11-14", "buy")


def test_paper_api(tmp_path):
    from fastapi.testclient import TestClient
    from agent.api import create_app

    path = tmp_path / "stan.db"
    history, broker = _history(), FakeAlpaca()
    book = PaperBook(path)
    _walk(broker, book, history, 45, 70)
    book.close()
    client = TestClient(create_app(path))
    data = client.get("/paper").json()
    assert data["enabled"] and data["approved_by"] == "Zy"
    assert data["start_equity"] == pytest.approx(100_000)
    assert data["kill_floor"] == pytest.approx(80_000)
    assert data["log"] and data["equity_history"]
    empty = TestClient(create_app(tmp_path / "none.db")).get("/paper").json()
    assert empty["enabled"] is False


def test_kill_switch_resells_a_pending_exit_whose_order_it_cancelled():
    history, broker, book = _history(), FakeAlpaca(), PaperBook(":memory:")
    _walk(broker, book, history, 45, 56)
    t = book.trades(["open"])[0]
    broker.submit_open_order(t["symbol"], t["qty"], "sell", "old-exit")   # an exit still waiting for the open
    book.update(t["symbol"], t["signal_date"], status="pending_exit", exit_order="old-exit", exit_reason="rsi")
    broker.cash -= 30_000
    days = history["SPY"].index
    cut = {s: df[df.index <= days[56]] for s, df in history.items()}
    run_evening(broker, book, cut, now=_evening(days[56]))
    assert broker.orders["old-exit"]["status"] == "canceled"
    after = book.trades(["pending_exit"])[0]
    assert after["exit_reason"] == "kill switch" and after["exit_order"] != "old-exit"
    new = broker.orders[after["exit_order"]]
    assert new["side"] == "sell" and new["qty"] == t["qty"] and new["status"] == "accepted"
