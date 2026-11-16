"""The QQQ trend bot rebuilt from the reel: mechanics, honesty on random data, and that it can see a real trend."""
import gzip
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from agent.stan import daytrade, qqq_trend as q

FIXTURE = Path(__file__).parent / "fixtures" / "day_trader" / "TSLA-60d5m.json.gz"


def _session(day: str, rets, start=100.0):
    bars, px = [], start
    for i, r in enumerate(rets):
        o, c = px, px * (1 + r)
        bars.append({"date": day, "slot": i, "o": o, "h": max(o, c) * 1.0004, "l": min(o, c) * 0.9996, "c": c, "v": 1000})
        px = c
    return bars


def trending_market(symbols=("QQQ", "SPY", "IWM", "DIA", "XLK"), days=60, seed=3):
    """Each day picks a direction and drifts that way all day: the market a trend bot is built for."""
    rng = np.random.default_rng(seed)
    out, d0 = {}, date(2026, 6, 1)
    for sym in symbols:
        sess, k = [], 0
        while len(sess) < days:
            d = d0 + timedelta(days=k)
            k += 1
            if d.weekday() >= 5:
                continue
            side = rng.choice([-1, 1])
            rets = side * 0.0006 + rng.normal(0, 0.0012, daytrade.SESSION_BARS)
            sess.append((d.isoformat(), _session(d.isoformat(), rets)))
        out[sym] = sess
    return out


def test_steady_uptrend_goes_long_and_rides_the_trail():
    bars = _session("2026-06-01", [0.0] * 10 + [0.001] * 68)
    trades = q.session_trades("2026-06-01", bars)
    assert trades and trades[0]["side"] == "long"
    assert trades[0]["R"] > 1  # profit lock + trail let a clean trend run past 1R


def test_steady_downtrend_goes_short():
    bars = _session("2026-06-01", [0.0] * 10 + [-0.001] * 68)
    trades = q.session_trades("2026-06-01", bars)
    assert trades and trades[0]["side"] == "short" and trades[0]["R"] > 1


def test_daily_limits_and_flat_by_close():
    sess = trending_market(days=20)
    for sym, days in sess.items():
        for d, bars in days:
            ts = q.session_trades(d, bars)
            assert len(ts) <= q.MAX_TRADES
            assert all(t["exit_bar"] <= daytrade.LAST_BAR and t["entry_bar"] >= q.START_BAR for t in ts)
            r = 0.0
            for t in ts:
                assert r > q.DAY_STOP_R  # no new trade once the day is down 2R
                r += t["R"]


def test_stop_never_loses_much_more_than_1r_without_a_gap():
    sess = trending_market(days=30, seed=9)
    trades = q.all_trades(sess, {})
    assert min(t["R"] for t in trades) >= -1.2


def test_exam_finds_a_real_trend_on_both_sides():
    exam = q.trend_exam(trending_market(), luck_reps=150)
    keys = {c["key"]: c["passed"] for c in exam["checks"]}
    assert keys["basic"] and keys["both_sides"] and keys["luck"], exam["checks"]
    assert exam["checks_total"] == 9


def test_exam_finds_nothing_on_random_walk_demo_data():
    sess = daytrade.exam_sessions(daytrade.load_charts(["QQQ", "SPY", "IWM", "AAPL", "MSFT"], demo=True))
    exam = q.trend_exam(sess, luck_reps=100)
    assert exam["verdict"] == "not_proven"
    assert exam["score"]["avg_R"] < 0  # costs eat a rule with no edge


def test_runs_on_real_bars_from_the_skill_pack():
    text = gzip.open(FIXTURE, "rt").read()
    sess = daytrade.exam_sessions({"TSLA": text})
    trades = q.all_trades(sess, {})
    assert trades
    assert {t["side"] for t in trades} <= {"long", "short"}
    assert all(t["risk"] > 0 for t in trades)
