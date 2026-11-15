"""Stan's use of the Day Trader skill pack. Offline: the pack's own TSLA fixture plus made-up data."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from agent.stan import daytrade as dt
from agent.stan.skills import SKILLS, library

FIXTURE = Path(__file__).parent / "fixtures" / "day_trader" / "TSLA-60d5m.json.gz"


@pytest.fixture(scope="module")
def tsla():
    return dt.fetch_chart_text("TSLA", FIXTURE)


def test_the_pack_is_vendored_unchanged_with_its_license():
    root = dt.SKILL_DIR
    assert (root / "LICENSE").read_text().startswith("MIT License")
    assert "3d0f4ef" in (root / "NOTICE.md").read_text()
    for key in (s["key"] for s in SKILLS):
        assert (root / key / "SKILL.md").exists()


def test_exam_matches_the_packs_published_history(tsla):
    # The pack's README: on TSLA's 59 sessions before Oct 19, the rule fired 33 times,
    # 13 hit the target, 15 the stop, 5 neither.
    sessions = dt.full_sessions(dt.sessions_from_text(tsla, "TSLA", datetime(2026, 10, 20)), before="2026-10-19")
    trades = [t for d, bars in sessions if (t := dt.orb_trade(d, bars, 1.0))]
    assert len(sessions) == 59 and len(trades) == 33
    outcomes = [t["outcome"] for t in trades]
    assert (outcomes.count("target"), outcomes.count("stop"), outcomes.count("close")) == (13, 15, 5)


def test_costs_are_charged_in_units_of_risk():
    t = {"R": 1.0, "entry": 100.0, "risk": 1.0}
    assert dt.net_r(t, 0.0005) == pytest.approx(1.0 - 2 * 0.0005 * 100)  # 0.10R for a 1% stop


def test_practice_walks_the_whole_chain(tsla):
    row = dt.practice_symbol("TSLA", tsla, "2026-10-15", dt.PaperRules(), replays=2000)
    assert row["entry_time"] == "10:45 AM" and row["stop"] < row["entry"] < row["target"]
    assert 0 <= row["p_target"] <= 1 and row["verdict"]
    assert row["hist_triggered"] is not None
    # One TSLA share costs more than the $50 plan allows: the budget manager refuses it.
    assert row["budget_qty"] == 0 and row["risk_verdict"] == "REFUSED"
    assert row["outcome"] in {"target", "stop", "close"}


def test_unfinished_days_are_not_practiced(tsla):
    assert dt.practice_symbol("TSLA", tsla, "2026-10-19", dt.PaperRules(), replays=500) is None


def test_the_time_machine_never_sees_the_practice_day(tsla):
    row = dt.practice_symbol("TSLA", tsla, "2026-10-15", dt.PaperRules(), replays=500)
    tm = dt.timemachine()
    sess = dt.full_sessions(dt.sessions_from_text(tsla, "TSLA", datetime(2026, 10, 20)), before="2026-10-15")
    assert sess[-1][0] < "2026-10-15"  # the pool ends before the day being graded
    assert row["hist_triggered"] == tm.history_replay(
        sess, {"setup": "orb", "or_minutes": 15, "stop_rule": "or_mid", "target_r": 1.0}, dt.LAST_BAR)["triggered"]


def test_budget_and_risk_pass_a_cheap_stock():
    plan = {"created_by": "trader-exit", "symbol": "F", "side": "buy", "date": "2026-10-19", "setup": "orb",
            "entry": 12.40, "entry_time": "10:15 AM", "entry_bar": 8, "interval": "5m",
            "opening_range": {"high": 12.35, "low": 12.05, "minutes": 15}, "stop": 12.20, "stop_rule": "or_mid",
            "target": 12.60, "target_r": 1.0, "risk_per_share": 0.2, "reward_risk": 1.0, "flat_by": "15:55"}
    br = dt.budget_and_risk(plan, dt.PaperRules(), "2026-10-19 10:15")
    assert br["qty"] == 3 and br["risk_verdict"] == "PASS"  # 75% position cap of $50 = 3 shares


def test_day_limits_take_at_most_three_trades_in_trigger_order():
    rows = [{"entry_time": t, "risk_verdict": "PASS", "budget_qty": 1, "entry": 10.0, "stop": 9.0,
             "risk_reason": "", "taken": 0} for t in ("10:30 AM", "9:50 AM", "11:00 AM", "10:00 AM")]
    rows = dt.apply_day_limits(rows, dt.PaperRules())
    taken = sorted(r["entry_time"] for r in rows if r["taken"])
    assert taken == sorted(["9:50 AM", "10:00 AM", "10:30 AM"])
    assert "3 trades" in next(r for r in rows if not r["taken"])["risk_reason"]


def test_exam_reports_seven_checks_on_demo_data():
    texts = dt.load_charts(["SPY", "AAA", "BBB", "CCC"], demo=True)
    exam = dt.day_trade_exam(dt.exam_sessions(texts), ["SPY"], luck_reps=50)
    assert exam["checks_total"] == 7 and exam["verdict"] in {"real_edge", "promising", "not_proven"}
    assert exam["score"]["trades"] > 0 and exam["unit"] == "R"


def test_practice_day_saves_once_and_reports(tmp_path):
    book = dt.PracticeBook(tmp_path / "stan.db")
    day = date.today()
    texts = dt.load_charts(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"], demo=True)
    from agent.stan.shadow import ShadowBook  # noqa: F401  (same file, separate tables)
    lines = []
    import pandas as pd
    last = pd.bdate_range(end=pd.Timestamp(day), periods=1)[0].date().isoformat()
    rows = dt.practice_day(book, texts, last, replays=500, log=lambda k, t: lines.append(t))
    assert rows and lines
    assert dt.practice_day(book, texts, last, replays=500) == []  # never twice
    stats = dt.practice_stats(book.rows())
    assert stats["setups"] == len(rows) and stats["days"] == 1
    assert 0 <= stats["tm_predicted_target"] <= 1


def test_skill_library():
    lib = library()
    assert lib["counts"] == {"in_use": 6, "learned": 4, "locked": 1}
