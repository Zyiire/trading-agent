"""Level play: volume profile, FVGs, signals without lookahead, the exam, and the live runner."""
import dataclasses
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from agent.stan import daytrade, levels as L
from agent.stan.intraday import LiveLevels, run_session, sessions_from_frame
from agent.stan.paper import PaperBook, set_strategy
from agent.stan.price_action.series import from_sessions
from agent.stan.shadow import ET
from tests.test_paper import FakeAlpaca

CFG = L.LevelConfig()


def test_profile_finds_poc_value_area_and_a_low_volume_node():
    # lots of trading around 100 and 104, almost none at 102: 102 is the low-volume node
    h = np.array([100.2] * 40 + [102.1] * 2 + [104.2] * 40)
    l = np.array([99.8] * 40 + [101.9] * 2 + [103.8] * 40)
    v = np.array([1000.0] * 40 + [50.0] * 2 + [900.0] * 40)
    lv = dict((n, p) for n, p in L.profile_levels(h, l, v, CFG) if n != "LVN")
    lvns = [p for n, p in L.profile_levels(h, l, v, CFG) if n == "LVN"]
    assert lv["POC"] == pytest.approx(100, abs=0.3)
    assert any(abs(p - 102) < 1.0 for p in lvns)
    assert lv["VAL"] <= lv["POC"] <= lv["VAH"]


def test_fvg_definition():
    h = np.array([10.0, 10.0, 9.0, 8.0]); l = np.array([9.5, 9.0, 8.0, 7.0])
    assert L.fvg(h, l, 3, -1, 3)            # bar 3's high (8.0) is below bar 1's low (9.0)
    assert not L.fvg(h, l, 3, 1, 3)


def _sessions(days=6, seed=2, level_cross_day=None):
    rng = np.random.default_rng(seed)
    out, px = [], 100.0
    d0 = datetime(2026, 11, 2)
    for k in range(days):
        bars = []
        for slot in range(78):
            o = px
            px *= 1 + rng.normal(0, 0.0015)
            bars.append({"slot": slot, "o": o, "h": max(o, px) * 1.0005, "l": min(o, px) * 0.9995, "c": px, "v": 1000})
        out.append(((d0 + timedelta(days=k)).date().isoformat(), bars))
    return out


def test_signals_ignore_the_future():
    s = from_sessions("QQQ", _sessions(days=8))
    lv_full = L.session_levels(s, CFG)
    for i in range(78 * 4, len(s), 7):
        cut = s.head(i + 1)
        lv_cut = L.session_levels(cut, CFG)
        d = int(s.day_idx[i])
        assert lv_cut[d] == lv_full[d]
        assert L.signal_at(cut, i, lv_cut[d], CFG) == L.signal_at(s, i, lv_full[d], CFG)


def test_exam_finds_nothing_on_random_walk_demo_data():
    sess = daytrade.exam_sessions(daytrade.load_charts(["QQQ", "SPY", "AAPL", "MSFT", "AMD"], demo=True))
    exam = L.levels_exam(sess, luck_reps=60)
    assert exam["checks_total"] == 9 and exam["verdict"] == "not_proven"


# ---------------------------------------------------------------- live runner
def _frame(sessions):
    rows, idx = [], []
    for d, bars in sessions:
        base = pd.Timestamp(d + " 09:30", tz=ET)
        for b in bars:
            idx.append((base + pd.Timedelta(minutes=5 * b["slot"])).tz_convert("UTC"))
            rows.append({"open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"], "volume": b["v"]})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _crossing_day():
    """Three quiet sessions around 100 and 104 (102 becomes a low-volume node), then a day that drops
    hard through 102 with a gap: a 'lose the level' short."""
    sess = []
    for k, mid in enumerate([100, 104, 100]):
        bars = [{"slot": i, "o": mid, "h": mid + 0.2, "l": mid - 0.2, "c": mid, "v": 1000} for i in range(78)]
        sess.append((f"2026-11-1{k}", bars))
    today = []
    for i in range(78):
        p = 103.0 if i < 5 else (101.0 - 0.05 * (i - 5) if i >= 7 else {5: 102.5, 6: 101.6}[i])
        today.append({"slot": i, "o": p + 0.05, "h": p + 0.1, "l": p - 0.1, "c": p, "v": 1000})
    today[5].update(o=103.0, h=103.05, l=102.4, c=102.5)
    today[6].update(o=102.1, h=102.2, l=101.5, c=101.6)   # closes below 102, and its high is below bar 4's low
    sess.append(("2026-11-13", today))
    return sess


def test_sessions_from_frame_drops_bars_still_forming():
    sess = _crossing_day()
    now = datetime(2026, 11, 13, 10, 2, tzinfo=ET)      # the 9:55 bar is still open
    got = sessions_from_frame(_frame(sess), now)
    assert got[-1][0] == "2026-11-13" and len(got[-1][1]) == 6   # 9:30 ... 9:50 bars have closed


def test_live_runner_sees_the_lost_level_and_respects_cash_accounts():
    sess = _crossing_day()
    now = datetime(2026, 11, 13, 10, 7, tzinfo=ET)       # the 10:00-10:05 bar, which breaks 102, has closed
    broker, book = FakeAlpaca(cash=100_000), PaperBook(":memory:")
    broker.mark({"QQQ": 101.6})
    out = LiveLevels(broker, book, lambda s: _frame(sess)).step(now)
    assert any(n == "LVN" and 101.5 < p < 102.5 for n, p in out["levels"])
    assert out["position"] is None and not broker.orders            # the fake broker is a cash account: no shorts
    assert any("cash account" in a for a in out["actions"])


def test_live_runner_shorts_on_a_margin_account():
    sess = _crossing_day()
    broker, book = FakeAlpaca(cash=100_000), PaperBook(":memory:")
    plain = broker.get_account
    broker.get_account = lambda: dataclasses.replace(plain(), multiplier="2")   # a margin account
    broker.mark({"QQQ": 101.6})
    out = LiveLevels(broker, book, lambda s: _frame(sess)).step(datetime(2026, 11, 13, 10, 7, tzinfo=ET))
    assert out["position"] and out["position"]["side"] == -1
    order = next(iter(broker.orders.values()))
    assert order["side"] == "sell" and order["symbol"] == "QQQ"
    assert out["position"]["qty"] * out["position"]["risk"] <= 100_000 * 0.01 + 1e-6   # 1% risk


def test_runner_exits_at_the_stop():
    broker, book = FakeAlpaca(cash=100_000), PaperBook(":memory:")
    broker.positions["QQQ"] = 10
    broker.mark({"QQQ": 101.0})
    book.set("levels_pos", {"side": 1, "qty": 10, "entry": 102.0, "stop": 101.5, "target": 104.0, "risk": 0.5,
                            "be": False, "day": "2026-11-13", "level": "LVN"})
    book.set("levels_day", "2026-11-13")
    sess = _crossing_day()
    out = LiveLevels(broker, book, lambda s: _frame(sess)).step(datetime(2026, 11, 13, 10, 7, tzinfo=ET))
    assert out["position"] is None
    assert any(o["symbol"] == "QQQ" and o["side"] == "sell" for o in broker.orders.values())


def test_one_bar_is_handled_once():
    broker, book = FakeAlpaca(cash=100_000), PaperBook(":memory:")
    sess = _crossing_day()
    r = LiveLevels(broker, book, lambda s: _frame(sess))
    now = datetime(2026, 11, 13, 10, 7, tzinfo=ET)
    r.step(now)
    n = len(broker.orders)
    r.step(now)
    assert len(broker.orders) == n


def test_run_session_stops_at_350_and_yields_to_another_runner():
    broker, book = FakeAlpaca(cash=100_000), PaperBook(":memory:")
    sess = _crossing_day()
    t = [datetime(2026, 11, 13, 15, 40, tzinfo=ET)]
    book.set("levels_heartbeat", {"who": "web", "at": t[0].timestamp()})
    slept = []

    def sleep(sec):
        slept.append(sec)
        t[0] += timedelta(seconds=sec)

    run_session(broker, book, lambda s: _frame(sess), now_fn=lambda: t[0], sleep=sleep, me="scheduled")
    assert t[0].time() >= datetime(2026, 11, 13, 15, 50).time()
    assert not broker.orders                      # the web app was trading, so the scheduled copy waited


def test_level_play_is_a_choice():
    book = PaperBook(":memory:")
    assert "levels" in set_strategy(book, ":memory:", "levels")
    assert book.get("strategy") == "levels"


def test_switching_to_level_play_sells_the_old_plan(tmp_path):
    from agent.stan.paper import run_evening
    from tests.test_tiers import _market
    h, broker, book = _market(), FakeAlpaca(cash=10_000), PaperBook(":memory:")
    broker.positions["SPY"] = 5
    days = h["SPY"].index
    cut = {k: df[df.index <= days[260]] for k, df in h.items()}
    broker.mark({k: float(df["close"].iloc[-1]) for k, df in cut.items()})
    book.set("strategy", "levels")
    sess = _crossing_day()
    now = datetime.combine(days[260].date(), datetime.min.time(), tzinfo=ET).replace(hour=17)
    run_evening(broker, book, cut, now=now, level_bars=lambda s: _frame(sess))
    assert any(o["symbol"] == "SPY" and o["side"] == "sell" for o in broker.orders.values())


def test_switching_away_from_level_play_closes_qqq(tmp_path):
    from agent.stan.paper import run_evening
    from tests.test_tiers import _market
    h, broker, book = _market(), FakeAlpaca(cash=10_000), PaperBook(":memory:")
    broker.positions["QQQ"] = 3
    book.set("levels_pos", {"side": 1, "qty": 3, "entry": 400.0, "stop": 398.0, "target": 404.0, "risk": 2.0,
                            "be": False, "day": "2026-01-01", "level": "LVN"})
    book.set("strategy", "medium")
    days = h["SPY"].index
    cut = {k: df[df.index <= days[260]] for k, df in h.items()}
    broker.mark({k: float(df["close"].iloc[-1]) for k, df in cut.items()})
    now = datetime.combine(days[260].date(), datetime.min.time(), tzinfo=ET).replace(hour=17)
    run_evening(broker, book, cut, now=now)
    assert any(o["symbol"] == "QQQ" and o["side"] == "sell" for o in broker.orders.values())
    assert book.get("levels_pos") is None
