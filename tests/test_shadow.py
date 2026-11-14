"""Shadow trading, run day by day on made-up data. No keys or network needed."""
from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.api import create_app
from agent.stan.shadow import (
    ET,
    ShadowBook,
    call_stats,
    complete_bars,
    make_calls,
    rule_positions,
    run_day,
    snapshot,
)
from agent.stan.strategies import SelloffRule
from agent.stan.train import demo_history

SYMBOLS = ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK"]


def _bars(closes, opens=None, start="2026-01-01"):
    days = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=days, dtype=float)
    o = pd.Series(opens if opens is not None else closes, index=days, dtype=float)
    return pd.DataFrame({"open": o, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6})


def test_unfinished_day_is_dropped_before_430pm():
    bars = _bars([1, 2, 3], start="2026-11-11")  # Wed, Thu, Fri Nov 13
    before = datetime(2026, 11, 13, 15, 0, tzinfo=ET)
    after = datetime(2026, 11, 13, 16, 45, tzinfo=ET)
    assert len(complete_bars({"X": bars}, before)["X"]) == 2
    assert len(complete_bars({"X": bars}, after)["X"]) == 3


def test_rule_positions_pending_open_and_closed():
    falls = [100 - i for i in range(20)]  # RSI hits 0
    rule = SelloffRule()
    # Signal on the very last day: pending, buy tomorrow.
    pend = rule_positions(_bars(falls), "X", since=_bars(falls).index[-1].date(), rule=rule)
    assert [p["status"] for p in pend] == ["pending"]
    # Signal then a few more falling days: open, marked to market.
    bars = _bars(falls + [79, 78])
    since = bars.index[19].date()
    opened = rule_positions(bars, "X", since=since, rule=rule)
    assert opened[0]["status"] == "open" and opened[0]["entry_price"] == pytest.approx(79)
    assert opened[0]["exit_price"] is None
    # Then a sharp rebound: RSI back above 50, closed at that close.
    bars = _bars(falls + [79, 78, 95, 99])
    closed = rule_positions(bars, "X", since=since, rule=rule)[0]
    assert closed["status"] == "closed"
    assert closed["exit_price"] in (95, 99)
    assert closed["net_ret"] == pytest.approx(closed["exit_price"] / 79 - 1 - 2 * 0.001)


def test_signals_before_the_start_date_are_ignored():
    falls = [100 - i for i in range(25)]
    bars = _bars(falls)
    later = bars.index[-1].date() + pd.Timedelta(days=1)
    assert rule_positions(bars, "X", since=later) == []


def test_calls_are_always_balanced():
    history = demo_history(SYMBOLS, years=3, seed=2)
    _, calls = make_calls(history, picks=5, max_iter=20)
    beats = [c for c in calls if c["call"] == "beat"]
    lags = [c for c in calls if c["call"] == "lag"]
    assert len(beats) == len(lags) == 5
    assert "SPY" not in {c["symbol"] for c in calls}  # the market is the yardstick, not a pick
    assert min(c["p_beat"] for c in beats) >= max(c["p_beat"] for c in lags)


def test_call_grading_and_stats():
    calls = [
        {"made_on": "2026-11-12", "symbol": "A", "call": "beat", "graded": 1, "graded_on": "2026-11-13",
         "stock_ret": 0.02, "market_ret": 0.01, "correct": 1, "p_beat": 0.6},
        {"made_on": "2026-11-12", "symbol": "B", "call": "lag", "graded": 1, "graded_on": "2026-11-13",
         "stock_ret": 0.03, "market_ret": 0.01, "correct": 0, "p_beat": 0.4},
    ]
    s = call_stats(calls)
    assert s["graded"] == 2 and s["hit_rate"] == 0.5
    assert s["beat_hit_rate"] == 1.0 and s["lag_hit_rate"] == 0.0 and s["balanced_hit_rate"] == 0.5
    assert s["always_beat_rate"] == 1.0  # both stocks beat the market that day
    assert s["avg_daily_spread"] == pytest.approx(0.01 - 0.02)
    assert s["beat_share_recent"] == 0.5


@pytest.fixture(scope="module")
def shadowed(tmp_path_factory):
    """Replay 15 trading days as if Stan ran every evening."""
    path = tmp_path_factory.mktemp("shadow") / "stan.db"
    book = ShadowBook(path)
    full = demo_history(SYMBOLS, years=3, seed=4)
    days = full["SPY"].index
    results = []
    for k in range(len(days) - 15, len(days) + 1):
        cut = {s: df[df.index <= days[k - 1]] for s, df in full.items()}
        when = datetime.combine(days[k - 1].date(), time(17, 0), tzinfo=ET)
        results.append(run_day(book, cut, now=when, demo=True, max_iter=20, train_years=1,
                               expected={"win_rate": 0.6, "avg": 0.005, "profit_factor": 1.3, "trades": 3000}))
    # Running the same evening twice must not duplicate anything.
    calls_before = len(book.calls())
    run_day(book, cut, now=when, demo=True, max_iter=20, train_years=1)
    assert len(book.calls()) == calls_before
    yield path, book, results, days
    book.close()


def test_shadow_replay_tracks_calls_and_positions(shadowed):
    path, book, results, days = shadowed
    started = book.get("started")
    assert started > days[-17].date().isoformat()  # nothing before day one counts
    calls = book.calls()
    assert len({c["made_on"] for c in calls}) == len(results)
    assert all(sum(c["call"] == "beat" for c in calls if c["made_on"] == d) == 5
               for d in {c["made_on"] for c in calls})
    graded = [c for c in calls if c["graded"]]
    assert len(graded) == 10 * (len(results) - 1)  # every day's calls graded except the newest
    for p in book.positions():
        assert p["signal_date"] >= started
        if p["status"] == "closed":
            assert p["exit_date"] >= p["entry_date"] and p["net_ret"] is not None
    kinds = {l["kind"] for l in book.logs(500)}
    assert {"milestone", "thought", "progress"} <= kinds


def test_closed_trades_are_never_rewritten(shadowed):
    path, book, _, _ = shadowed
    closed = [p for p in book.positions() if p["status"] == "closed"]
    if not closed:
        pytest.skip("no closed trade in this replay")
    p = dict(closed[0])
    book.save_position({**p, "net_ret": 9.9, "status": "closed"})
    assert book.position(p["symbol"], p["signal_date"])["net_ret"] == p["net_ret"]


def test_shadow_api(shadowed):
    path, book, _, _ = shadowed
    client = TestClient(create_app(path))
    data = client.get("/shadow").json()
    assert data["started"] == book.get("started")
    assert data["mode"] == "demo"
    assert len(data["calls"]["latest"]) == 10
    assert data["calls"]["stats"]["graded"] > 0
    assert set(data["selloff"]["stats"]) >= {"open", "closed", "win_rate", "needed"}
    assert data == snapshot(book)


def test_shadow_api_before_any_run(tmp_path):
    client = TestClient(create_app(tmp_path / "empty.db"))
    data = client.get("/shadow").json()
    assert data["started"] is None and data["calls"]["latest"] == []


def test_school_rerun_does_not_wipe_shadow_history(shadowed):
    from agent.stan.journal import Journal

    path, book, _, _ = shadowed
    before = len(book.calls())
    j = Journal(path)
    j.reset()
    j.close()
    assert len(book.calls()) == before
