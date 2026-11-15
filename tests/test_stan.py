"""Stan's school, end to end, on made-up data. No keys or network needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.api import create_app
from agent.stan.journal import Journal
from agent.stan.persona import rank_for
from agent.stan.strategies import SelloffRule, score, selloff_trades, stress_test
from agent.stan.study import FEATURES, StudyConfig, build_panel, find_lessons, grade_year, luck_test, make_features, max_drawdown
from agent.stan.train import CURRICULUM, demo_history, run_training

FAST = StudyConfig(max_iter=30, luck_reps=100)


def test_features_do_not_peek_at_the_future():
    bars = demo_history(["SPY"], years=3)["SPY"]
    full = make_features(bars)
    cut = make_features(bars.iloc[:400])
    # Features for day 399 must be identical whether or not later days exist.
    pd.testing.assert_series_equal(full[FEATURES].iloc[399], cut[FEATURES].iloc[399])
    # The outcome column is the only one allowed to use tomorrow.
    assert np.isnan(cut["next_ret"].iloc[-1])
    assert full["next_ret"].iloc[10] == pytest.approx(bars["close"].iloc[11] / bars["close"].iloc[10] - 1)


def _year(p_up, next_ret, symbol="AAA"):
    days = pd.bdate_range("2020-01-01", periods=len(p_up))
    return pd.DataFrame({"date": days, "symbol": symbol, "p_up": p_up, "next_ret": next_ret,
                         "ma200": 0.1, "rsi14": 50.0, "vol_ratio": 1.0})


def test_grading_charges_costs_and_counts_trades():
    cfg = StudyConfig(threshold=0.55, cost_per_side=0.01)
    # Long for two days (+10%, +10%), flat for one, long one day (-5%).
    exam, trades, daily = grade_year(_year([0.9, 0.9, 0.1, 0.9], [0.10, 0.10, 0.02, -0.05]), cfg)
    assert exam["trades"] == 2 and len(trades) == 2
    assert trades[0]["ret"] == pytest.approx(1.1 * 1.1 - 1 - 0.02)
    assert trades[1]["ret"] == pytest.approx(-0.05 - 0.02)
    assert exam["win_rate"] == 0.5 and exam["passed"] is True
    # Daily: buy (+10% - 1%), hold (+10%), sell (-1%), buy and close at year end (-5% - 1% - 1%).
    expected = (1.10 - 0.01) * 1.10 * (1 - 0.01) * (1 - 0.05 - 0.02) - 1
    assert exam["stan_return"] == pytest.approx(expected, abs=1e-4)
    assert exam["hold_return"] == pytest.approx(1.1 * 1.1 * 1.02 * 0.95 - 1, abs=1e-4)
    assert exam["accuracy"] == 0.5  # right, right, wrong, wrong
    assert exam["exposure"] == 0.75
    assert exam["max_dd"] < 0 and exam["hold_max_dd"] < 0
    assert len(daily["stan"]) == 4


def test_no_trades_means_no_pass():
    exam, trades, _ = grade_year(_year([0.5, 0.5], [0.01, 0.01]), StudyConfig())
    assert trades == [] and exam["passed"] is False and exam["stan_return"] == 0.0


def _t(ret, up=True, rsi_=50.0):
    return {"ret": ret, "above_ma200": up, "rsi14": rsi_, "vol_ratio": 1.0, "confidence": 0.6}


def test_lessons_need_enough_trades():
    assert find_lessons([_t(-0.01, up=False)] * 5) == []
    many = [_t(0.02, up=True)] * 40 + [_t(-0.02, up=False)] * 40
    kinds = {l["kind"]: l["title"] for l in find_lessons(many)}
    assert "downtrend" in kinds["mistake"] and "uptrend" in kinds["strength"]


def test_a_profitable_condition_close_to_average_is_not_a_mistake():
    # Shaped like the real run: uptrend trades win 59.1% vs 59.3% overall and make money.
    trades = ([_t(0.02, True)] * 1397 + [_t(-0.02, True)] * 967
              + [_t(0.02, False, 25)] * 193 + [_t(-0.02, False, 25)] * 96)
    lessons = find_lessons(trades)
    assert all(l["kind"] != "mistake" for l in lessons)
    assert any(l["kind"] == "strength" and "selloff" in l["title"] for l in lessons)


def test_beating_a_losing_average_is_not_a_mistake():
    # Everything loses money, but the downtrend bucket loses less than average.
    trades = [_t(-0.03, True)] * 60 + [_t(-0.01, False)] * 60
    titles = [l["title"] for l in find_lessons(trades) if l["kind"] == "mistake"]
    assert titles == ["I do worse in an uptrend (price above its 200-day average)"]


def test_max_drawdown():
    assert max_drawdown(pd.Series([0.10, -0.50, 0.20])) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


# ---------- selloff strategy ----------
def _bars(closes, opens=None):
    days = pd.bdate_range("2020-01-01", periods=len(closes))
    c = pd.Series(closes, index=days, dtype=float)
    o = pd.Series(opens if opens is not None else closes, index=days, dtype=float)
    return pd.DataFrame({"open": o, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6})


def test_selloff_rule_enters_after_signal_and_exits_on_recovery():
    # 20 steady falls push RSI to 0, then a sharp rebound pushes it above 50.
    closes = [100 - i for i in range(21)] + [95, 96]
    opens = list(closes)
    opens[21] = 81.0  # the morning after the last fall
    bars = _bars(closes, opens)
    rule = SelloffRule(entry_rsi=30, exit_rsi=50, max_days=10, entry="next_open")
    trades = selloff_trades(bars, "X", rule)
    first = trades[0]
    assert first["entry"] == bars.index[14]  # first day RSI exists and is under 30
    # next open after the signal day is bars.open[15] == closes[15]
    assert first["gross"] == pytest.approx(bars["close"].iloc[first["j"]] / bars["open"].iloc[15] - 1)
    assert first["days"] <= 10
    close_rule = SelloffRule(entry="close")
    t2 = selloff_trades(bars, "X", close_rule)[0]
    assert t2["gross"] == pytest.approx(bars["close"].iloc[t2["j"]] / bars["close"].iloc[14] - 1)


def test_selloff_max_days_caps_the_hold():
    closes = [100 - i for i in range(40)]  # never recovers
    trades = selloff_trades(_bars(closes), "X", SelloffRule(max_days=5))
    assert trades and all(t["days"] <= 5 for t in trades)


def test_score_counts_costs():
    trades = [{"gross": 0.02}, {"gross": -0.01}]
    s = score(trades, 0.001)
    assert s["trades"] == 2 and s["win_rate"] == 0.5
    assert s["avg"] == pytest.approx((0.018 - 0.012) / 2)
    assert s["profit_factor"] == pytest.approx(0.018 / 0.012, abs=1e-3)


def test_luck_test_does_not_reward_random_trades():
    history = demo_history(["AAA", "BBB"], years=4, seed=3)
    closes = {k: v["close"] for k, v in history.items()}
    rng = np.random.default_rng(0)
    trades = []
    for sym, c in closes.items():
        for i in rng.integers(0, len(c) - 10, size=150):
            trades.append({"symbol": sym, "entry": c.index[i], "days": 3,
                           "ret": float(c.iloc[i + 3] / c.iloc[i] - 1)})
    luck = luck_test(trades, closes, cost=0.0, reps=300)
    assert luck["p_value"] > 0.01  # random trades should look like luck
    # A cheat that knows the future should look clearly better than luck.
    cheat = [dict(t, ret=abs(t["ret"])) for t in trades]
    assert luck_test(cheat, closes, cost=0.0, reps=300)["p_value"] < 0.01


def test_stress_test_runs_every_check():
    history = demo_history(["SPY", "QQQ", "AAA", "BBB"], years=6, seed=5)
    seen = []
    result = stress_test(history, ["SPY", "QQQ"], on_check=lambda c: seen.append(c["key"]), luck_reps=100)
    assert seen == ["basic", "years", "luck", "costs", "timing", "grid", "new_stocks", "regime",
                    "per_day", "drawdown"]
    assert result["checks_total"] == 10 and 0 <= result["checks_passed"] <= 10
    assert result["verdict"] in {"real_edge", "promising", "not_proven"}
    assert result["versus"]["exposure"] > 0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    path = tmp_path_factory.mktemp("stan") / "stan.db"
    journal = Journal(path)
    from agent.stan.daytrade import load_charts

    summary = run_training(journal, demo_history(["SPY", "QQQ", "AAPL"], years=8), "practice data", demo=True,
                           cfg=FAST, intraday=load_charts(["SPY", "QQQ", "AAPL"], demo=True))
    yield path, journal, summary
    journal.close()


def test_training_runs_every_task(trained):
    _, journal, summary = trained
    tasks = journal.tasks()
    assert [t["key"] for t in tasks] == [k for k, _ in CURRICULUM]
    assert all(t["status"] == "done" and t["progress"] == 1.0 for t in tasks)
    exams = journal.exams()
    assert summary["years_tested"] == len(exams) >= 3
    assert min(e["year"] for e in exams) >= demo_history(["SPY"], years=8)["SPY"].index.year.min() + 3
    assert 0.3 < summary["accuracy"] < 0.7
    meta = journal.meta()
    assert meta["mode"] == "demo" and meta["state"] == "finished"
    assert meta["stage"] == ("paper_candidate" if summary["ready_for_paper"] else "student")
    kinds = {t["kind"] for t in journal.thoughts(limit=1000)}
    assert {"milestone", "progress", "thought"} <= kinds


def test_training_without_intraday_data_skips_only_the_day_exam():
    journal = Journal(":memory:")
    run_training(journal, demo_history(["SPY", "QQQ"], years=6), "practice data", demo=True, cfg=FAST)
    status = {t["key"]: t["status"] for t in journal.tasks()}
    assert status["daytrade"] == "failed" and status["report"] == "done"


def test_training_without_data_fails_clearly():
    journal = Journal(":memory:")
    assert run_training(journal, {}, "nowhere") == {}
    assert journal.tasks()[0]["status"] == "failed"
    assert journal.meta()["state"] == "stopped"


def test_api_serves_the_journal(trained):
    path, journal, summary = trained
    client = TestClient(create_app(path))
    stan = client.get("/stan").json()
    assert stan["name"] == "Stan" and stan["mode"] == "demo" and stan["state"] == "finished"
    assert stan["tasks_done"] == stan["tasks_total"] == len(CURRICULUM)
    assert stan["strategy"]["checks_total"] == 10
    assert stan["daytrade_exam"]["checks_total"] == 7
    skills = client.get("/skills").json()
    assert len(skills["skills"]) == 11 and skills["pack"]["license"] == "MIT"
    assert next(k for k in skills["skills"] if k["key"] == "trader-broker")["status"] == "locked"
    assert stan["summary"]["trades"] == summary["trades"]
    assert len(client.get("/tasks").json()) == len(CURRICULUM)
    thoughts = client.get("/thoughts", params={"limit": 500}).json()
    assert thoughts == sorted(thoughts, key=lambda t: t["id"])
    last = thoughts[-1]["id"]
    assert client.get("/thoughts", params={"after": last}).json() == []
    assert len(client.get("/exams").json()) == summary["years_tested"]
    assert isinstance(client.get("/lessons").json(), list)


def test_api_token(trained):
    path, _, _ = trained
    client = TestClient(create_app(path, token="secret"))
    assert client.get("/health").status_code == 200
    assert client.get("/stan").status_code == 401
    assert client.get("/stan", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_ranks():
    assert rank_for("student")["level"] == 1 and rank_for("student")["next"]
    assert rank_for("top_tier")["next"] is None
    assert rank_for("nonsense")["key"] == "student"
