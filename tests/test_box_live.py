"""Live box-breakout trading against a fake Alpaca paper account: one bracket a day, the flatten job,
the `box` paper strategy, and the paper-only broker methods."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent.broker.alpaca import AlpacaBroker
from agent.broker.base import AccountInfo, MarketClock, OrderInfo, PositionInfo
from agent.config import Settings
from agent.stan import paper as P
from agent.stan.box import live as L
from agent.stan.box import rules
from agent.stan.paper import PaperBook, run_evening, set_strategy
from agent.stan.shadow import ET

ROOT = Path(__file__).resolve().parents[1]
DAY = date(2026, 11, 19)          # a Thursday
CID = "stan-box-20261119"


def at(hh, mm, ss=0, d=DAY):
    return datetime.combine(d, time(hh, mm, ss), tzinfo=ET)


# ---------------------------------------------------------------- synthetic minutes
def frame(start: datetime, closes, spread=0.05) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(pd.Timestamp(start), periods=len(closes), freq="1min")
    return pd.DataFrame({"o": closes, "h": closes + spread, "l": closes - spread, "c": closes}, index=idx)


def history(trend="up", days=5) -> pd.DataFrame:
    """Five earlier days of 4:00-20:00 minutes that rise (or fall) steadily: a clear 15-minute trend."""
    parts, n = [], 16 * 60
    for k in range(days, 0, -1):
        d = DAY - timedelta(days=k)
        i = days - k
        lo, hi = (50 + i * 10, 60 + i * 10) if trend == "up" else (160 - i * 10, 150 - i * 10)
        parts.append(frame(at(4, 0, d=d), np.linspace(lo, hi, n)))
    return pd.concat(parts)


def today(breakout="long", minutes_after=15, gap=None) -> pd.DataFrame:
    """9:00-9:29 inside a 100-101 box, then flat at 100.5, then a 5-minute bar from 9:40 closing outside."""
    pre = frame(at(9, 0), [100.5] * 30, spread=0.0)
    pre.loc[pre.index[3], "h"] = 101.0
    pre.loc[pre.index[17], "l"] = 100.0
    if gap is not None:                     # drop a whole five-minute bucket of the box
        pre = pre[(pre.index < at(9, gap)) | (pre.index >= at(9, gap + 5))]
    calm = frame(at(9, 30), [100.5] * 10, spread=0.1)
    if breakout is None:
        move = frame(at(9, 40), [100.5] * (minutes_after + 46), spread=0.1)
    else:
        end = 102.0 if breakout == "long" else 99.0
        move = frame(at(9, 40), list(np.linspace(100.6 if breakout == "long" else 100.4, end, 5))
                     + [end] * minutes_after, spread=0.1)
    return pd.concat([pre, calm, move])


def upto(df, now):
    return df[df.index + pd.Timedelta(minutes=1) <= pd.Timestamp(now)]


# ---------------------------------------------------------------- the fake broker
class FakeBroker:
    def __init__(self, equity=100_000.0, buying_power=200_000.0, paper=True):
        self.equity, self.buying_power, self.paper = equity, buying_power, paper
        self.orders: dict[str, dict] = {}
        self.sent: list[tuple] = []
        self.closes: list[tuple] = []
        self.cancelled: list[str] = []
        self.positions: dict[str, float] = {}
        self.mark = 0.0
        self.close_fills = True
        self.close = time(16, 0)      # today's close; time(13, 0) on a half day
        self.close_errors = 0         # close_position raises this many times first
        self.on_close_error = None    # called on each such error (e.g. a leg fills meanwhile)
        self.is_open = True           # False: a holiday

    def get_clock(self):
        when = datetime.combine(DAY, self.close, tzinfo=ET)
        return MarketClock(when, self.is_open, when + timedelta(days=1), when)

    def get_account(self):
        return AccountInfo("a", "ACTIVE", self.paper, self.equity, self.equity, self.buying_power, None, None,
                           False, False, "4")

    def get_positions(self):
        return [PositionInfo(s, "us_equity", q, 0, None, None) for s, q in self.positions.items() if q]

    def submit_bracket_order(self, symbol, qty, side, stop, target, client_order_id):
        if client_order_id in self.orders:
            raise RuntimeError("client_order_id must be unique")
        self.sent.append((symbol, qty, side, stop, target, client_order_id))
        exit_side = "sell" if side == "buy" else "buy"
        self.orders[client_order_id] = {
            "symbol": symbol, "qty": qty, "side": side, "status": "accepted", "filled_qty": 0.0, "price": None,
            "type": "market",
            "legs": [{"cid": client_order_id + "-tp", "type": "limit", "side": exit_side, "qty": qty,
                      "status": "held", "filled_qty": 0.0, "price": None, "limit": round(target, 2)},
                     {"cid": client_order_id + "-sl", "type": "stop", "side": exit_side, "qty": qty,
                      "status": "held", "filled_qty": 0.0, "price": None, "stop": round(stop, 2)}]}
        return self.get_order(client_order_id)

    # test helpers
    def fill_entry(self, cid, price):
        o = self.orders[cid]
        o.update(status="filled", filled_qty=float(o["qty"]), price=price)
        for leg in o["legs"]:
            leg["status"] = "new"
        self.positions[o["symbol"]] = self.positions.get(o["symbol"], 0) + (o["qty"] if o["side"] == "buy" else -o["qty"])

    def hit(self, cid, kind, price):
        o = self.orders[cid]
        for leg in o["legs"]:
            if leg["type"] == kind:
                leg.update(status="filled", filled_qty=float(leg["qty"]), price=price)
            else:
                leg["status"] = "canceled"
        self.positions[o["symbol"]] -= o["qty"] if o["side"] == "buy" else -o["qty"]

    @staticmethod
    def _info(cid, o):
        return OrderInfo(cid, o.get("symbol", "QQQ"), o["side"], o["qty"], o["status"], o["filled_qty"], o["price"],
                         None, order_type=o.get("type"), limit_price=o.get("limit"), stop_price=o.get("stop"))

    def get_order(self, cid):
        o = self.orders.get(cid)
        return None if o is None else self._info(cid, o)

    def get_bracket(self, cid):
        o = self.orders.get(cid)
        if o is None:
            return None, []
        return self._info(cid, o), [self._info(leg["cid"], {**leg, "symbol": o["symbol"]}) for leg in o["legs"]]

    def cancel_orders(self, prefix):
        n = 0
        for cid, o in self.orders.items():
            if not cid.startswith(prefix):
                continue
            for x in [o, *o.get("legs", [])]:
                if x["status"] not in ("filled", "canceled"):
                    x["status"] = "canceled"
                    self.cancelled.append(x.get("cid", cid))
                    n += 1
        return n

    def close_position(self, symbol, qty, side, client_order_id):
        if self.close_errors:
            self.close_errors -= 1
            if self.on_close_error:
                self.on_close_error()
            raise RuntimeError("insufficient qty available for order")
        self.closes.append((symbol, qty, side, client_order_id))
        o = {"symbol": symbol, "qty": qty, "side": side, "status": "accepted", "filled_qty": 0.0, "price": None,
             "type": "market"}
        if self.close_fills:
            o.update(status="filled", filled_qty=float(qty), price=self.mark)
            self.positions[symbol] = self.positions.get(symbol, 0) + (qty if side == "buy" else -qty)
        self.orders[client_order_id] = o
        return self._info(client_order_id, o)


@pytest.fixture
def book():
    b = PaperBook(":memory:")
    b.set("strategy", "box")
    yield b
    b.close()


def step(broker, book, now, minutes=None, hist=None, dry_run=False):
    minutes = today() if minutes is None else minutes
    hist15 = rules.bars(history() if hist is None else hist, 15)
    return L.step(broker, book, hist15, upto(minutes, now), now, dry_run=dry_run, log=lambda text: None)


def texts(book):
    return " | ".join(r["text"] for r in book.logs(50))


# ---------------------------------------------------------------- watch: the entry
def test_watch_sends_one_bracket_on_the_breakout():
    broker, book = FakeBroker(), PaperBook(":memory:")
    book.set("strategy", "box")
    assert step(broker, book, at(9, 44, 5)) is False          # 9:40 bar not finished: keep watching
    assert broker.sent == []
    assert step(broker, book, at(9, 45, 5)) is True
    assert broker.sent == [("QQQ", 500, "buy", 100.0, 108.0, CID)]   # 1% of $100k / $2 risk = 500 shares
    state = book.get("box_today")
    assert state["day"] == "2026-11-19" and state["status"] == "sent"
    assert state["box"] == [101.0, 100.0] and state["order"]["id"] == CID and state["order"]["qty"] == 500
    assert "bracket" in texts(book)


def test_second_watch_the_same_day_never_sends_again(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5)) is True
    assert step(broker, book, at(9, 46, 5)) is True and len(broker.sent) == 1
    fresh = PaperBook(":memory:")                     # the book lost its state (crash before saving)
    fresh.set("strategy", "box")
    assert step(broker, fresh, at(9, 47, 5)) is True
    assert len(broker.sent) == 1                      # Alpaca already has today's id
    assert fresh.get("box_today")["status"] == "sent"


def test_send_error_but_alpaca_took_it_counts_as_sent(book):
    broker = FakeBroker()
    real = broker.submit_bracket_order

    def flaky(*a, **k):
        real(*a, **k)
        raise TimeoutError("read timed out")
    broker.submit_bracket_order = flaky
    assert step(broker, book, at(9, 45, 5)) is True
    assert book.get("box_today")["status"] == "sent" and len(broker.sent) == 1


def test_dry_run_sends_and_saves_nothing(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5), dry_run=True) is True
    assert broker.sent == [] and book.get("box_today") is None and book.logs() == []


def test_watch_loop_exits_once_the_trade_is_placed(book):
    broker, minutes = FakeBroker(), today()
    clock = iter([at(9, 31, 5), at(9, 32, 5), at(9, 45, 5), at(9, 46, 5)])
    naps = []
    state = L.watch(broker, book, rules.bars(history(), 15), lambda now: upto(minutes, now),
                    clock=lambda: next(clock), sleep=naps.append, log=lambda text: None)
    assert state["status"] == "sent" and len(broker.sent) == 1 and len(naps) == 2


def test_watch_loop_ends_at_1031_with_no_trade(book):
    broker, minutes = FakeBroker(), today(breakout=None)
    times = [at(10, 29, 5), at(10, 31, 5)]
    clock = iter(times)
    state = L.watch(broker, book, rules.bars(history(), 15), lambda now: upto(minutes, now),
                    clock=lambda: next(clock), sleep=lambda s: None, log=lambda text: None)
    assert state["status"] == "no_signal" and broker.sent == []


def test_short_breakout_in_a_downtrend(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5), minutes=today("short"), hist=history("down")) is True
    assert broker.sent == [("QQQ", 500, "sell", 101.0, 93.0, CID)]    # risk $2 a share, target 3R lower


def test_trend_filter_blocks_a_long_in_a_downtrend(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5), hist=history("down")) is True
    assert broker.sent == []
    assert book.get("box_today")["status"] == "blocked" and "trend" in book.get("box_today")["reason"]


@pytest.mark.parametrize("gap", [0, 10, 25])
def test_incomplete_box_means_no_trade(book, gap):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5), minutes=today(gap=gap)) is True
    assert broker.sent == [] and book.get("box_today")["status"] == "no_box"


def test_holiday_with_no_bars_means_no_trade(book):
    broker = FakeBroker()
    broker.is_open = False
    empty = today().iloc[:0]
    assert step(broker, book, at(9, 31, 5), minutes=empty) is False    # bars may still be on their way
    assert step(broker, book, at(9, 35, 5), minutes=empty) is True
    assert broker.sent == [] and book.get("box_today")["status"] == "no_box"


def test_no_breakout_by_1030(book):
    broker = FakeBroker()
    assert step(broker, book, at(10, 29, 5), minutes=today(breakout=None)) is False
    assert step(broker, book, at(10, 30, 5), minutes=today(breakout=None)) is True
    assert book.get("box_today")["status"] == "no_signal"


def test_breakout_seen_too_late_after_a_restart_is_skipped(book):
    broker = FakeBroker()
    assert step(broker, book, at(10, 10, 5), minutes=today(minutes_after=40)) is True
    assert broker.sent == [] and book.get("box_today")["status"] == "skipped"
    assert "late" in book.get("box_today")["reason"]


def test_too_little_trend_history_is_skipped(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5), hist=history(days=1)) is True
    assert broker.sent == [] and "history" in book.get("box_today")["reason"]


@pytest.mark.parametrize("setup,word", [
    (lambda b: b.set("halted", True), "paused"),
    (lambda b: b.set("strategy", "medium"), "medium"),
    (lambda b: b.set("box_results", [{"day": "2026-11-16", "r": -1.0}, {"day": "2026-11-17", "r": -1.1},
                                     {"day": "2026-11-18", "r": -1.0}]), "3R"),
])
def test_skips(book, setup, word):
    setup(book)
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5)) is True
    assert broker.sent == []
    state = book.get("box_today")
    assert state["status"] == "skipped" and word in state["reason"]
    assert word in texts(book)


def test_losses_from_last_week_do_not_count(book):
    book.set("box_results", [{"day": "2026-11-12", "r": -1.0}, {"day": "2026-11-13", "r": -1.0},
                             {"day": "2026-11-16", "r": -1.0}, {"day": "2026-11-17", "r": -1.0}])
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5)) is True and len(broker.sent) == 1


def test_not_enough_buying_power_sizes_down_and_logs_the_smaller_risk(book):
    broker = FakeBroker(buying_power=20_000.0)       # 500 shares at $102 = $51,000; 95% of $20k buys 186
    assert step(broker, book, at(9, 45, 5)) is True
    assert broker.sent == [("QQQ", 186, "buy", 100.0, 108.0, CID)]
    state = book.get("box_today")
    assert state["status"] == "sent" and state["order"]["risk_pct"] == pytest.approx(186 * 2 / 100_000)
    assert "0.37% of $100,000.00, not 1%" in texts(book) and "sized down to 186 shares" in texts(book)


def test_not_even_one_share_of_buying_power_is_skipped_with_a_reason(book):
    broker = FakeBroker(buying_power=50.0)
    assert step(broker, book, at(9, 45, 5)) is True
    assert broker.sent == []
    state = book.get("box_today")
    assert state["status"] == "skipped" and "buying power" in state["reason"]
    assert "buying power" in texts(book)


def test_under_one_share_is_skipped(book):
    broker = FakeBroker(equity=100.0)                # $1 of risk, $2 a share
    assert step(broker, book, at(9, 45, 5)) is True
    assert broker.sent == [] and "share" in book.get("box_today")["reason"]


# ---------------------------------------------------------------- flatten
def sent_book(broker, book):
    assert step(broker, book, at(9, 45, 5)) is True
    broker.fill_entry(CID, 102.10)
    return book


def test_flatten_cancels_closes_and_records_r(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.mark = 104.10
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert set(broker.cancelled) == {CID + "-tp", CID + "-sl"}
    assert broker.closes == [("QQQ", 500, "sell", CID + "-flat1")]
    assert state["status"] == "closed"
    res = state["result"]
    assert res["reason"] == "time" and res["entry"] == 102.10 and res["exit"] == 104.10
    assert res["r"] == pytest.approx(1.0) and res["pnl"] == pytest.approx(1000.0)
    assert book.get("box_results") == [res]
    assert "about a minute before" in texts(book)
    # a second flatten does nothing
    L.flatten(broker, book, at(15, 56, 5), sleep=lambda s: None, log=lambda text: None)
    assert len(broker.closes) == 1 and len(book.get("box_results")) == 1


def test_flatten_after_the_stop_already_hit_records_r_from_the_fill(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.hit(CID, "stop", 99.90)
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == []
    assert state["status"] == "closed" and state["result"]["reason"] == "stop"
    assert state["result"]["r"] == pytest.approx(-1.1) and state["result"]["exit"] == 99.90


def test_flatten_after_the_target_hit(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.hit(CID, "limit", 108.0)
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [] and state["result"]["reason"] == "target"
    assert state["result"]["r"] == pytest.approx(2.95)


def test_flatten_short_position(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5), minutes=today("short"), hist=history("down")) is True
    broker.fill_entry(CID, 99.0)
    broker.mark = 98.0
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [("QQQ", 500, "buy", CID + "-flat1")]
    assert state["result"]["r"] == pytest.approx(0.5) and state["result"]["pnl"] == pytest.approx(500.0)


def test_flatten_that_does_not_fill_is_retried(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.close_fills, broker.mark = False, 103.10
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, wait=3, log=lambda text: None,
                      attempts=1)
    assert state["status"] == "flatten_failed" and book.get("box_results") in (None, [])
    assert L.held_qty(broker, book) == 500
    broker.close_fills = True
    state = L.flatten(broker, book, at(15, 58, 5), sleep=lambda s: None, log=lambda text: None)
    assert CID + "-flat1" in broker.cancelled                  # the stuck close was cancelled first
    assert broker.closes[-1] == ("QQQ", 500, "sell", CID + "-flat2")
    assert state["status"] == "closed" and state["result"]["r"] == pytest.approx(0.5)


def test_flatten_with_nothing_open_sends_nothing(book):
    broker = FakeBroker()
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [] and broker.cancelled == [] and state is None
    book.set("box_today", {"day": "2026-11-19", "status": "no_signal", "reason": "nothing broke out"})
    L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [] and broker.cancelled == []


def test_flatten_when_the_entry_never_filled(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5)) is True
    broker.orders[CID]["status"] = "rejected"
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [] and state["status"] == "closed" and state["result"]["r"] is None


def test_flatten_dry_run_sends_nothing(book):
    broker = FakeBroker()
    sent_book(broker, book)
    L.flatten(broker, book, at(15, 55, 5), dry_run=True, sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [] and broker.cancelled == [] and book.get("box_today")["status"] == "sent"


def test_next_morning_closes_a_trade_left_open_first(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.mark = 103.10                                # flatten never ran yesterday (PC asleep)
    nxt = DAY + timedelta(days=1)
    nxt_minutes = today().copy()
    nxt_minutes.index = nxt_minutes.index + pd.Timedelta(days=1)
    hist = pd.concat([history(), today()])
    now = at(9, 45, 5, d=nxt)
    assert L.step(broker, book, rules.bars(hist, 15), upto(nxt_minutes, now), now,
                  log=lambda text: None, sleep=lambda s: None) is True
    assert broker.closes == [("QQQ", 500, "sell", CID + "-flat1")]
    assert book.get("box_results")[0]["day"] == "2026-11-19"
    assert broker.sent[-1][-1] == "stan-box-20261120"


# ---------------------------------------------------------------- the paper account on `box`
class TierFake(FakeBroker):
    """FakeBroker plus the market-on-open orders the evening run uses."""

    def submit_fractional_order(self, symbol, qty, side, client_order_id):
        if client_order_id in self.orders:
            raise RuntimeError("client_order_id must be unique")
        self.orders[client_order_id] = {"symbol": symbol, "qty": qty, "side": side, "status": "accepted",
                                        "filled_qty": 0.0, "price": None}
        return self.get_order(client_order_id)

    submit_open_order = submit_fractional_order

    def cancel_open_orders(self):
        return 0


def daily_history(day=DAY):
    days = pd.bdate_range(end=pd.Timestamp(day), periods=260)
    out = {}
    for sym, base in {"SPY": 500.0, "QQQ": 400.0, "TQQQ": 50.0}.items():
        c = pd.Series(base * 1.0005 ** np.arange(260), index=days)
        out[sym] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6})
    return out


def evening(broker, book, day=DAY):
    return run_evening(broker, book, daily_history(day), now=datetime.combine(day, time(17, 0), tzinfo=ET))


def test_box_strategy_sells_tier_holdings_once(book, tmp_path):
    broker = TierFake()
    broker.positions = {"SPY": 12.5, "TQQQ": 30.0}
    book.set("strategy", "medium")
    set_strategy(book, str(tmp_path / "j.db"), "box")
    r = evening(broker, book)
    sells = sorted((o["symbol"], o["side"], o["qty"]) for o in broker.orders.values())
    assert sells == [("SPY", "sell", 12.5), ("TQQQ", "sell", 30.0)]
    assert r["strategy"] == "box" and book.get("box_tiers_sold") == "2026-11-19"
    broker.positions = {"SPY": 1.0}                    # e.g. a dividend or a leftover: not re-sold daily
    evening(broker, book, DAY + timedelta(days=1))
    assert len(broker.orders) == 2


def test_box_strategy_places_no_tier_or_selloff_orders(book):
    broker = TierFake()
    r = evening(broker, book)
    assert broker.orders == {} and r["strategy"] == "box" and not r["halted"]


def test_switching_back_to_box_sells_again(book, tmp_path):
    broker = TierFake()
    broker.positions = {"SPY": 5.0}
    evening(broker, book)
    set_strategy(book, str(tmp_path / "j.db"), "medium")
    set_strategy(book, str(tmp_path / "j.db"), "box")
    broker.positions = {"SPY": 7.0}
    evening(broker, book, DAY + timedelta(days=1))
    assert sorted(o["qty"] for o in broker.orders.values()) == [5.0, 7.0]


def test_reconcile_ignores_the_box_position(book):
    broker = TierFake()
    assert step(broker, book, at(9, 45, 5)) is True
    broker.fill_entry(CID, 102.1)
    book.add_trade({"symbol": "QQQ", "signal_date": "2026-11-10", "status": "open", "qty": 3, "entry_order": "x"})
    broker.positions["QQQ"] += 3                       # a selloff QQQ trade on top of the box position
    r = evening(broker, book)
    assert not r["halted"] and not book.get("halted")
    assert L.held_qty(broker, book) == 500


def test_kill_switch_also_closes_the_box_position(book):
    broker = TierFake()
    assert step(broker, book, at(9, 45, 5)) is True
    broker.fill_entry(CID, 102.1)
    broker.mark = 101.0
    book.set("start_equity", 200_000.0)               # equity $100k is 50% below the start
    r = evening(broker, book)
    assert r["halted"] and broker.closes and broker.closes[0][:3] == ("QQQ", 500, "sell")
    assert book.get("box_today")["status"] == "closed"


def test_set_strategy_and_cli_accept_box(book, tmp_path):
    book.set("strategy", "medium")
    assert "box" in set_strategy(book, str(tmp_path / "j.db"), "box")
    assert book.get("strategy") == "box"
    journal = tmp_path / "stan.db"
    assert P.main(["--journal", str(journal), "--use", "box"]) == 0
    b = PaperBook(journal)
    assert b.get("strategy") == "box"
    b.close()


# ---------------------------------------------------------------- the Alpaca broker methods
class FakeTrading:
    def __init__(self):
        self.submitted, self.cancelled = [], []
        t = datetime(2026, 11, 19, 14, 45, tzinfo=timezone.utc)

        def order(cid, oid, status, typ, legs=None, qty="500", filled="0", price=None):
            return SimpleNamespace(id=oid, client_order_id=cid, symbol="QQQ", side=SimpleNamespace(value="buy"),
                                   qty=qty, status=SimpleNamespace(value=status), filled_qty=filled,
                                   filled_avg_price=price, filled_at=t, submitted_at=t,
                                   order_type=SimpleNamespace(value=typ), legs=legs)
        self.legs = [order("tp", "L1", "new", "limit"), order("sl", "L2", "held", "stop")]
        self.parent = order(CID, "P1", "filled", "market", legs=self.legs, filled="500", price="102.1")
        self.other = order("stan-SPY-20261119-buy", "O1", "new", "market")
        self.position = SimpleNamespace(symbol="QQQ", qty="500", side=SimpleNamespace(value="long"))

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id="N", client_order_id=req.client_order_id, symbol=req.symbol, side=req.side,
                               qty=req.qty, status="accepted", filled_qty="0", filled_avg_price=None,
                               filled_at=None, submitted_at=None, order_type=SimpleNamespace(value="market"))

    def get_orders(self, req):
        self.orders_request = req
        return [self.parent, self.other]

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)

    def get_order_by_client_id(self, cid):
        if cid != CID:
            raise RuntimeError("404 order not found")
        return SimpleNamespace(**{**vars(self.parent), "legs": None})

    def get_order_by_id(self, oid, filter=None):
        self.by_id = (oid, filter)
        return self.parent

    def get_open_position(self, symbol):
        if symbol != "QQQ":
            raise RuntimeError("404 position does not exist")
        return self.position


def alpaca(trading):
    return AlpacaBroker(Settings(api_key="k", secret_key="s"), trading, object(), object())


def test_submit_bracket_order_builds_the_request():
    trading = FakeTrading()
    alpaca(trading).submit_bracket_order("QQQ", 500, "buy", 100.004, 108.0049, CID)
    req = trading.submitted[0].to_request_fields()
    assert req["order_class"].value == "bracket" and req["time_in_force"].value == "day"
    assert req["qty"] == 500 and req["side"].value == "buy" and req["client_order_id"] == CID
    assert req["take_profit"] == {"limit_price": 108.0} and req["stop_loss"] == {"stop_price": 100.0}


def test_bracket_order_needs_whole_shares():
    with pytest.raises(ValueError):
        alpaca(FakeTrading()).submit_bracket_order("QQQ", 2.5, "buy", 1, 2, CID)


def test_cancel_orders_cancels_open_legs_with_the_prefix_only():
    trading = FakeTrading()
    assert alpaca(trading).cancel_orders("stan-box-") == 2
    assert trading.cancelled == ["L1", "L2"]                # the filled parent and the SPY order stay
    assert trading.orders_request.nested is True
    with pytest.raises(ValueError):
        alpaca(trading).cancel_orders("")


def test_get_bracket_reads_the_legs():
    trading = FakeTrading()
    trading.legs[0].limit_price, trading.legs[1].stop_price = "108", "100"
    parent, legs = alpaca(trading).get_bracket(CID)
    assert parent.filled_avg_price == 102.1 and parent.order_type == "market"
    assert [x.order_type for x in legs] == ["limit", "stop"]
    assert legs[0].limit_price == 108.0 and legs[1].stop_price == 100.0 and parent.limit_price is None
    assert trading.by_id[0] == "P1" and trading.by_id[1].nested is True
    assert alpaca(trading).get_bracket("stan-box-nope") == (None, [])


def test_close_position_sends_a_market_order_on_the_side_given():
    trading = FakeTrading()
    o = alpaca(trading).close_position("QQQ", 500, "sell", CID + "-flat1")
    req = trading.submitted[0].to_request_fields()
    assert req["side"].value == "sell" and req["qty"] == 500 and req["client_order_id"] == CID + "-flat1"
    assert req["time_in_force"].value == "day" and o.client_order_id == CID + "-flat1"
    alpaca(trading).close_position("QQQ", 300, "buy", CID + "-flat2")    # the account is net long: still buys
    assert trading.submitted[-1].side.value == "buy" and trading.submitted[-1].qty == 300
    with pytest.raises(ValueError):
        alpaca(trading).close_position("QQQ", 1, "short", CID + "-flat3")


@pytest.mark.parametrize("call", [
    lambda b: b.submit_bracket_order("QQQ", 1, "buy", 1.0, 2.0, CID),
    lambda b: b.cancel_orders("stan-box-"),
    lambda b: b.close_position("QQQ", 1, "sell", CID + "-flat1"),
])
def test_new_order_methods_refuse_a_live_account(call):
    trading = FakeTrading()
    b = alpaca(trading)
    b._paper = False
    with pytest.raises(RuntimeError):
        call(b)
    assert trading.submitted == [] and trading.cancelled == []


# ---------------------------------------------------------------- the scheduled jobs
@pytest.mark.parametrize("bat,job,cmd", [("run_box.bat", "box", "watch"), ("run_flatten.bat", "flatten", "flatten")])
def test_box_bat_files(bat, job, cmd):
    raw = (ROOT / bat).read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")
    text = raw.decode()
    log = f"data\\{job}.log"
    assert f"echo ==== %date% %time% ==== >> {log}" in text
    assert f"%PY% -m agent.stan.box.live {cmd} >> {log} 2>&1" in text
    assert 'set "CODE=%errorlevel%"' in text and f"echo Exit code: %CODE% >> {log}" in text
    assert f"%PY% -m agent.discord_bot.notify {job} %CODE% >> {log} 2>&1" in text


def test_schedule_adds_the_two_box_tasks():
    text = (ROOT / "schedule_trading.bat").read_text()
    assert '/TN "Stan box run" /TR "\\"%~dp0run_box.bat\\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:31' in text
    assert '/TN "Stan box flatten" /TR "\\"%~dp0run_flatten.bat\\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:55' in text
    assert 'schtasks /Delete /TN "Stan box run" /F' in text and 'schtasks /Delete /TN "Stan box flatten" /F' in text


def test_cli_has_dry_run_for_both_commands():
    parser = L.build_parser()
    assert parser.parse_args(["watch", "--dry-run"]).dry_run
    assert parser.parse_args(["flatten", "--dry-run"]).dry_run


# ---------------------------------------------------------------- fix round 1
def test_half_day_is_skipped(book):
    broker = FakeBroker()
    broker.close = time(13, 0)
    assert step(broker, book, at(9, 45, 5)) is True
    assert broker.sent == []
    state = book.get("box_today")
    assert state["status"] == "skipped" and state["reason"].startswith("half day: the bracket would expire at the early close")


def test_box_short_on_top_of_a_selloff_long_closes_by_buying_the_box_qty(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5), minutes=today("short"), hist=history("down")) is True
    broker.positions["QQQ"] = 800.0                     # a selloff QQQ long of 800 ...
    broker.fill_entry(CID, 99.0)                        # ... plus the box short of 500: net long 300
    broker.mark = 98.0
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [("QQQ", 500, "buy", CID + "-flat1")]
    assert broker.positions["QQQ"] == 800.0 and state["status"] == "closed"


def test_a_step_that_fails_once_is_retried_next_minute(book):
    broker, minutes = FakeBroker(), today()
    real, calls = broker.get_account, []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("Alpaca hiccup")
        return real()
    broker.get_account = flaky
    clock = iter([at(9, 45, 5), at(9, 46, 5), at(9, 47, 5)])
    lines = []
    state = L.watch(broker, book, rules.bars(history(), 15), lambda now: upto(minutes, now),
                    clock=lambda: next(clock), sleep=lambda s: None, log=lines.append)
    assert len(broker.sent) == 1 and state["status"] == "sent"
    assert any("Alpaca hiccup" in x for x in lines)


def test_watch_records_a_failure_if_it_never_decides(book):
    broker = FakeBroker()
    clock = iter([at(10, 31, 5)])

    def down(now):
        raise ConnectionError("no data")
    state = L.watch(broker, book, rules.bars(history(), 15), down, clock=lambda: next(clock),
                    sleep=lambda s: None, log=lambda text: None)
    assert state["status"] == "failed" and "no data" in state["reason"] and broker.sent == []


def test_restart_recovers_stop_target_and_risk_from_the_legs(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5)) is True
    fresh = PaperBook(":memory:")
    fresh.set("strategy", "box")
    assert step(broker, fresh, at(9, 47, 5)) is True and len(broker.sent) == 1
    order = fresh.get("box_today")["order"]
    assert (order["stop"], order["target"], order["risk"]) == (100.0, 108.0, 2.0)
    broker.fill_entry(CID, 102.1)
    broker.mark = 104.1
    state = L.flatten(broker, fresh, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert state["result"]["r"] == pytest.approx(1.0)
    assert L.weekly_r(fresh, DAY) == pytest.approx(1.0)


def test_restart_keeps_a_record_that_knows_more(book):
    broker = FakeBroker()
    assert step(broker, book, at(9, 45, 5)) is True
    state = {**book.get("box_today"), "status": "flatten_failed", "close_orders": [CID + "-flat1"]}
    book.set("box_today", state)
    assert step(broker, book, at(9, 50, 5)) is True
    assert book.get("box_today") == state and len(broker.sent) == 1


def test_close_retry_rereads_fills_so_it_never_over_closes(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.close_errors = 1
    broker.on_close_error = lambda: broker.hit(CID, "stop", 99.9)   # the stop fills while the close is refused
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [] and state["status"] == "closed" and state["result"]["reason"] == "stop"


def test_next_morning_cleanup_is_not_reported_as_a_355_close(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.mark = 103.1
    state = L.flatten(broker, book, at(9, 31, 5, d=DAY + timedelta(days=1)), sleep=lambda s: None,
                      log=lambda text: None)
    assert state["result"]["reason"] == "early" and "3:55" not in book.logs(1)[0]["text"]


def test_tier_holdings_are_retried_if_a_sell_fails(book):
    broker = TierFake()
    broker.positions = {"SPY": 5.0}
    real = broker.submit_fractional_order

    def refuse(*a, **k):
        raise RuntimeError("market closed")
    broker.submit_fractional_order = refuse
    evening(broker, book)
    assert book.get("box_tiers_sold") is None
    broker.submit_fractional_order = real
    evening(broker, book, DAY + timedelta(days=1))
    assert book.get("box_tiers_sold") == "2026-11-20" and len(broker.orders) == 1


def test_trend_history_skips_the_top_up_for_a_missing_or_stale_db(tmp_path):
    import sqlite3
    from agent.stan.box import data
    calls, start = [], at(9, 0)
    fake_backfill = lambda client, db, log=None: calls.append(db) or 0
    assert L.trend_history(None, tmp_path / "none.db", start, backfill=fake_backfill, log=lambda t: None).empty
    stale = tmp_path / "stale.db"
    with sqlite3.connect(stale) as db:
        db.execute(data.SCHEMA)
        db.execute("INSERT INTO bars_1m VALUES ('QQQ', ?, 1, 1, 1, 1, 1, 1, 'x')",
                   (int(datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc).timestamp()),))
    assert L.trend_history(None, stale, start, backfill=fake_backfill, log=lambda t: None).empty
    assert calls == []
    fresh = tmp_path / "fresh.db"
    with sqlite3.connect(fresh) as db:
        db.execute(data.SCHEMA)
        db.execute("INSERT INTO bars_1m VALUES ('QQQ', ?, 1, 1, 1, 1, 1, 1, 'x')",
                   (int(datetime(2026, 11, 18, 14, 0, tzinfo=timezone.utc).timestamp()),))
    hist = L.trend_history(None, fresh, start, backfill=fake_backfill, log=lambda t: None)
    assert calls == [fresh] and len(hist) == 1


# ---------------------------------------------------------------- final review fixes
def test_send_error_when_alpaca_cannot_be_asked_is_retried_not_failed(book):
    broker = FakeBroker()
    real_send, real_get = broker.submit_bracket_order, broker.get_order
    down = {"on": False}

    def send(*a, **k):
        real_send(*a, **k)                      # Alpaca took it, but the answer was lost ...
        down["on"] = True
        raise TimeoutError("read timed out")

    def get(cid):
        if down["on"]:
            down["on"] = False                  # ... and the follow-up check fails too
            raise ConnectionError("Alpaca down")
        return real_get(cid)
    broker.submit_bracket_order, broker.get_order = send, get
    with pytest.raises(TimeoutError):
        step(broker, book, at(9, 45, 5))
    assert book.get("box_today") is None        # not recorded as failed
    assert step(broker, book, at(9, 46, 5)) is True       # next minute: found at Alpaca, not sent again
    assert len(broker.sent) == 1 and book.get("box_today")["status"] == "sent"


def test_watch_retries_a_send_whose_outcome_is_unknown(book):
    broker, minutes = FakeBroker(), today()
    real_send, real_get = broker.submit_bracket_order, broker.get_order
    calls = []

    def send(*a, **k):
        calls.append(1)
        raise TimeoutError("read timed out")    # never reached Alpaca

    def get(cid):
        if len(calls) == 1 and not getattr(get, "failed", False):
            get.failed = True
            raise ConnectionError("Alpaca down")
        return real_get(cid)
    broker.submit_bracket_order, broker.get_order = send, get
    clock = iter([at(9, 45, 5), at(9, 46, 5)])
    state = L.watch(broker, book, rules.bars(history(), 15), lambda now: upto(minutes, now),
                    clock=lambda: next(clock), sleep=lambda s: None, log=lambda text: None)
    assert len(calls) == 2 and state["status"] == "failed"     # the second send failed plainly
    broker.submit_bracket_order = real_send


def test_flatten_finds_todays_order_the_book_never_recorded(book):
    broker = FakeBroker()
    broker.submit_bracket_order("QQQ", 500, "buy", 100.0, 108.0, CID)
    broker.fill_entry(CID, 102.1)
    book.set("box_today", {"day": "2026-11-19", "status": "failed", "reason": "the watch ended", "box": [101.0, 100.0]})
    broker.mark = 104.1
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [("QQQ", 500, "sell", CID + "-flat1")]
    assert state["status"] == "closed" and state["result"]["r"] == pytest.approx(1.0)
    assert state["box"] == [101.0, 100.0] and "found at Alpaca" in state["reason"]
    assert "is at Alpaca" in texts(book)
    L.flatten(broker, book, at(15, 56, 5), sleep=lambda s: None, log=lambda text: None)   # closed: left alone
    assert len(broker.closes) == 1 and len(book.get("box_results")) == 1


def test_flatten_with_an_empty_book_still_checks_alpaca(book):
    broker = FakeBroker()
    broker.submit_bracket_order("QQQ", 500, "sell", 101.0, 93.0, CID)
    broker.fill_entry(CID, 99.0)
    broker.mark = 98.0
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert broker.closes == [("QQQ", 500, "buy", CID + "-flat1")] and state["result"]["r"] == pytest.approx(0.5)


class SlowCancel(FakeBroker):
    """Cancelling takes a moment: legs sit in pending_cancel, and the stop fills before the cancel lands."""

    def __init__(self, fill_stop=True, **kw):
        super().__init__(**kw)
        self.fill_stop, self.reads = fill_stop, 0

    def cancel_orders(self, prefix):
        for o in self.orders.values():
            for leg in o.get("legs", []):
                if leg["status"] not in ("filled", "canceled"):
                    leg["status"] = "pending_cancel"
        return 2

    def get_bracket(self, cid):
        self.reads += 1
        o = self.orders.get(cid)
        if o and self.reads == 2 and self.fill_stop:
            self.hit(cid, "stop", 99.9)
        return super().get_bracket(cid)


def test_flatten_waits_for_the_legs_so_a_stop_filling_during_the_cancel_is_not_closed_twice(book):
    broker = SlowCancel()
    sent_book(broker, book)
    naps = []
    state = L.flatten(broker, book, at(15, 55, 5), sleep=naps.append, log=lambda text: None)
    assert broker.closes == [] and state["status"] == "closed" and state["result"]["reason"] == "stop"
    assert naps and naps[0] == L.SETTLE_NAP


def test_flatten_gives_up_if_the_legs_never_settle(book):
    broker = SlowCancel(fill_stop=False)
    sent_book(broker, book)
    naps = []
    state = L.flatten(broker, book, at(15, 55, 5), sleep=naps.append, log=lambda text: None, attempts=1)
    assert broker.closes == [] and state["status"] == "flatten_failed" and "10 seconds" in state["reason"]
    assert sum(naps) == pytest.approx(L.SETTLE_NAP * (L.SETTLE_POLLS - 1))


def test_flatten_retries_for_a_few_minutes_before_giving_up(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.close_fills, broker.mark = False, 103.1
    naps = []

    def nap(s):
        naps.append(s)
        if s == L.FLATTEN_GAP and naps.count(L.FLATTEN_GAP) == 2:
            broker.close_fills = True             # Alpaca recovers before the third attempt
    state = L.flatten(broker, book, at(15, 55, 5), sleep=nap, wait=2, log=lambda text: None)
    assert state["status"] == "closed" and state["result"]["r"] == pytest.approx(0.5)
    assert [c[-1] for c in broker.closes] == [CID + "-flat1", CID + "-flat2", CID + "-flat3"]
    assert "still open with no stop" not in texts(book)


def test_flatten_final_failure_says_the_position_has_no_stop(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.close_fills = False
    naps = []
    state = L.flatten(broker, book, at(15, 55, 5), sleep=naps.append, wait=1, log=lambda text: None, attempts=3)
    assert state["status"] == "flatten_failed" and len(broker.closes) == 3
    assert naps.count(L.FLATTEN_GAP) == 2
    assert texts(book).count("The position is still open with no stop") == 1
    assert L.FLATTEN_ATTEMPTS * L.FLATTEN_GAP >= 180


def test_flatten_retries_an_alpaca_error(book):
    broker = FakeBroker()
    sent_book(broker, book)
    broker.mark = 104.1
    real, calls = broker.cancel_orders, []

    def flaky(prefix):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("Alpaca hiccup")
        return real(prefix)
    broker.cancel_orders = flaky
    state = L.flatten(broker, book, at(15, 55, 5), sleep=lambda s: None, log=lambda text: None)
    assert state["status"] == "closed" and len(calls) == 2


def test_empty_iex_on_an_open_day_is_a_glitch_retried_next_minute(book):
    broker = FakeBroker()                         # the market is open: not a holiday
    empty = today().iloc[:0]
    assert step(broker, book, at(9, 35, 5), minutes=empty) is False
    assert step(broker, book, at(9, 40, 5), minutes=empty) is False
    assert book.get("box_today") is None
    assert step(broker, book, at(9, 45, 5)) is True and len(broker.sent) == 1


def test_kill_switch_landing_during_the_step_stops_the_send(book):
    broker = FakeBroker()
    real = broker.get_account

    def account():
        book.set("halted", True)                  # the 9:35 run's kill switch fires meanwhile
        book.set("halt_reason", "Kill switch")
        return real()
    broker.get_account = account
    assert step(broker, book, at(9, 45, 5)) is True
    assert broker.sent == []
    state = book.get("box_today")
    assert state["status"] == "skipped" and "paused just before the order went out (Kill switch)" in state["reason"]


def test_iex_box_gap_falls_back_to_sip_after_946(book):
    broker = FakeBroker()
    iex, sip = today(gap=10), today()
    sip_calls = []

    def fetch_sip():
        sip_calls.append(1)
        return sip[sip.index < at(9, 30)]
    hist15 = rules.bars(history(), 15)
    run = lambda now: L.step(broker, book, hist15, upto(iex, now), now, log=lambda text: None,
                             fetch_sip_box=fetch_sip)
    assert run(at(9, 45, 5)) is False and sip_calls == [] and book.get("box_today") is None
    assert run(at(9, 46, 5)) is True and sip_calls == [1]
    assert broker.sent == [("QQQ", 500, "buy", 100.0, 108.0, CID)]
    assert book.get("box_today")["box"] == [101.0, 100.0]


def test_box_gap_on_both_feeds_is_no_box(book):
    broker = FakeBroker()
    gappy = today(gap=10)
    now = at(9, 46, 5)
    assert L.step(broker, book, rules.bars(history(), 15), upto(gappy, now), now, log=lambda text: None,
                  fetch_sip_box=lambda: gappy[gappy.index < at(9, 30)]) is True
    state = book.get("box_today")
    assert state["status"] == "no_box" and "both feeds" in state["reason"] and broker.sent == []


def test_stale_trend_history_skips_the_day_with_a_reason(tmp_path, book):
    import sqlite3
    from agent.stan.box import data
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as con:
        con.execute(data.SCHEMA)
        con.execute("INSERT INTO bars_1m VALUES ('QQQ', ?, 1, 1, 1, 1, 1, 1, 'x')",
                    (int(datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc).timestamp()),))

    def broken(client, db, log=None):
        raise ConnectionError("SIP down")
    hist = L.trend_history(None, db, at(9, 0), backfill=broken, log=lambda t: None, prev_session=date(2026, 11, 18))
    assert "2026-11-17" in hist.attrs["problem"] and "SIP down" in hist.attrs["problem"]
    ok = L.trend_history(None, db, at(9, 0), backfill=lambda c, d, log=None: 0, log=lambda t: None,
                         prev_session=date(2026, 11, 17))
    assert "problem" not in ok.attrs
    hist15 = rules.bars(history(), 15)
    hist15.attrs["problem"] = hist.attrs["problem"]
    broker = FakeBroker()
    assert L.step(broker, book, hist15, upto(today(), at(9, 45, 5)), at(9, 45, 5), log=lambda t: None) is True
    state = book.get("box_today")
    assert broker.sent == [] and state["status"] == "skipped" and "stale" in state["reason"]


def test_previous_session_uses_the_calendar_or_the_last_weekday():
    assert L.previous_session(date(2026, 11, 30)) == date(2026, 11, 27)              # Monday -> Friday
    cal = lambda a, b: [date(2026, 11, 25), date(2026, 11, 27)]                      # Thanksgiving closed
    assert L.previous_session(date(2026, 11, 27), cal) == date(2026, 11, 25)
    assert L.previous_session(date(2026, 11, 27), lambda a, b: 1 / 0) == date(2026, 11, 26)


def test_reconcile_on_box_halts_on_qqq_nobody_owns(book):
    broker = TierFake()
    broker.positions = {"QQQ": 7.0}             # no box trade, no selloff trade: unexplained QQQ
    r = evening(broker, book)
    assert r["halted"] and "QQQ" in book.get("halt_reason")


def test_reconcile_on_box_halts_when_the_box_holds_less_than_alpaca_shows(book):
    broker = TierFake()
    assert step(broker, book, at(9, 45, 5)) is True
    broker.fill_entry(CID, 102.1)
    broker.positions["QQQ"] += 40                # an untracked 40 on top of the box's 500
    assert evening(broker, book)["halted"]


def test_reconcile_survives_the_box_entry_filling_mid_read(book):
    broker = TierFake()
    assert step(broker, book, at(9, 45, 5)) is True
    real = broker.get_positions

    def positions():
        if broker.orders[CID]["status"] != "filled":
            broker.fill_entry(CID, 102.1)        # fills after the box qty was read, before the positions
        return real()
    broker.get_positions = positions
    r = evening(broker, book)
    assert not r["halted"] and not book.get("halted")


def test_switch_to_box_keeps_spy_owned_by_open_selloff_trades(book, tmp_path):
    broker = TierFake()
    broker.positions = {"SPY": 10.0}
    book.set("strategy", "selloff")
    book.add_trade({"symbol": "SPY", "signal_date": "2026-11-12", "status": "open", "qty": 10, "entry_order": "x",
                    "entry_price": 500.0})
    set_strategy(book, str(tmp_path / "j.db"), "box")
    r = evening(broker, book)
    assert not r["halted"] and not any(k.startswith("stan-tier") for k in broker.orders)   # no tier sell of SPY
    assert book.get("box_tiers_sold") == "2026-11-19"


def test_sell_all_keeps_the_shares_given():
    from agent.stan import tiers
    broker, said = TierFake(), []
    broker.positions = {"SPY": 12.5, "TQQQ": 3.0}
    assert tiers.sell_all(broker, "2026-11-19", lambda k, t: said.append(t), keep={"SPY": 10})
    assert sorted((o["symbol"], o["qty"]) for o in broker.orders.values()) == [("SPY", 2.5), ("TQQQ", 3.0)]
    assert any("keeping the 10" in t for t in said)


# ---------------------------------------------------------------- live minute sources (Yahoo, IEX, SIP)
def gapped(df, gap):
    return df[(df.index < at(9, gap)) | (df.index >= at(9, gap + 5))]


def test_compose_uses_yahoo_when_its_box_is_complete_and_drops_the_forming_minute():
    full = today()
    calls = []
    yahoo = lambda now: (calls.append("Y"), full[full.index <= pd.Timestamp(now).floor("min")])[1]  # incl. forming
    iex = lambda now: (calls.append("I"), gapped(upto(full, now), 10))[1]
    now = at(9, 45, 5)
    out = L.compose_today([("Yahoo", yahoo), ("IEX", iex)], now, log=lambda t: None)
    assert out.attrs["source"] == "Yahoo"
    assert out.index.max() == at(9, 44)                       # the 9:45 minute is still forming: dropped
    assert L._box_buckets(rules.bars(out, 5)) == 6


def test_compose_falls_back_to_iex_when_yahoo_misses_a_bucket():
    full = today()
    yahoo = lambda now: gapped(upto(full, now), 15)
    iex = lambda now: upto(full, now)
    out = L.compose_today([("Yahoo", yahoo), ("IEX", iex)], at(9, 45, 5), log=lambda t: None)
    assert out.attrs["source"] == "IEX" and L._box_buckets(rules.bars(out, 5)) == 6


def test_compose_takes_signal_minutes_from_the_next_source_when_the_box_source_lags():
    full = today()
    yahoo = lambda now: full[full.index < at(9, 42)]          # stalled after 9:41
    iex = lambda now: gapped(upto(full, now), 0)
    out = L.compose_today([("Yahoo", yahoo), ("IEX", iex)], at(9, 45, 5), log=lambda t: None)
    assert out.attrs["source"] == "box from Yahoo, signal minutes from IEX"
    assert out.index.max() == at(9, 44) and L._box_buckets(rules.bars(out, 5)) == 6


def test_compose_survives_a_source_error():
    full = today()

    def broken(now):
        raise ConnectionError("Yahoo down")
    lines = []
    out = L.compose_today([("Yahoo", broken), ("IEX", lambda now: upto(full, now))], at(9, 45, 5), log=lines.append)
    assert out.attrs["source"] == "IEX" and any("Yahoo down" in x for x in lines)


def test_watch_with_both_sources_incomplete_waits_then_uses_sip_after_946(book):
    broker, full = FakeBroker(), today()
    lines = []
    minutes = L.TodayMinutes([("Yahoo", lambda now: gapped(upto(full, now), 10)),
                              ("IEX", lambda now: gapped(upto(full, now), 20))], log=lines.append)
    sip_calls = []
    sip = lambda: (sip_calls.append(1), full[full.index < at(9, 30)])[1]
    clock = iter([at(9, 45, 5), at(9, 46, 5)])
    state = L.watch(broker, book, rules.bars(history(), 15), minutes, clock=lambda: next(clock),
                    sleep=lambda s: None, log=lines.append, fetch_sip_box=sip)
    assert sip_calls == [1] and broker.sent == [("QQQ", 500, "buy", 100.0, 108.0, CID)]
    assert state["source"].startswith("SIP box, signal minutes from Yahoo")
    assert any("today's minutes come from Yahoo (box incomplete on every source)" in x for x in lines)


def test_watch_records_yahoo_as_the_source(book):
    broker, full = FakeBroker(), today()
    minutes = L.TodayMinutes([("Yahoo", lambda now: upto(full, now)), ("IEX", lambda now: full.iloc[:0])],
                             log=lambda t: None)
    clock = iter([at(9, 45, 5)])
    state = L.watch(broker, book, rules.bars(history(), 15), minutes, clock=lambda: next(clock),
                    sleep=lambda s: None, log=lambda t: None)
    assert state["status"] == "sent" and state["source"] == "Yahoo" and "Minutes from Yahoo." in texts(book)


def test_kill_switch_keeps_spy_owned_by_open_selloff_trades(book):
    broker = TierFake()
    broker.positions = {"SPY": 10.0, "TQQQ": 4.0}  # SPY all owned by the selloff trade; TQQQ by the tier
    book.add_trade({"symbol": "SPY", "signal_date": "2026-11-18", "status": "open", "qty": 10, "entry_order": "x",
                    "entry_price": 500.0})
    book.set("start_equity", 200_000.0)               # equity $100k: the kill switch fires
    book.set("strategy", "medium")
    r = evening(broker, book)
    assert r["halted"]
    spy = sorted(o["qty"] for o in broker.orders.values() if o["symbol"] == "SPY")
    assert spy == [10]                                # only the selloff trade's own sell, not 10 more from sell_all
    assert [o["qty"] for o in broker.orders.values() if o["symbol"] == "TQQQ"] == [4.0]


def test_fetch_yahoo_prepost_is_opt_in(monkeypatch):
    import yfinance as yf
    from agent.stan.zerodte import bars as B
    seen = []
    monkeypatch.setattr(yf, "download", lambda *a, **k: seen.append(k) or None)
    assert B.fetch_yahoo("QQQ") == [] and B.fetch_yahoo("QQQ", period="1d", prepost=True) == []
    assert (seen[0]["period"], seen[0]["prepost"]) == ("7d", False)
    assert (seen[1]["period"], seen[1]["prepost"]) == ("1d", True)


def test_flatten_retries_the_alpaca_lookup_when_it_is_unreachable(book):
    broker = FakeBroker()
    broker.submit_bracket_order("QQQ", 500, "buy", 100.0, 108.0, CID)
    broker.fill_entry(CID, 102.1)
    broker.mark = 104.1
    real, calls = broker.get_order, []

    def flaky(cid):
        calls.append(cid)
        if len(calls) == 1:
            raise ConnectionError("Alpaca down")
        return real(cid)
    broker.get_order = flaky
    naps = []
    state = L.flatten(broker, book, at(15, 55, 5), sleep=naps.append, log=lambda text: None)
    assert naps and state["status"] == "closed" and broker.closes == [("QQQ", 500, "sell", CID + "-flat1")]
