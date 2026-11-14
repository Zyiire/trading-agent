"""Stan's school, end to end, on made-up data. No keys or network needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.api import create_app
from agent.stan.journal import Journal
from agent.stan.persona import rank_for
from agent.stan.study import FEATURES, StudyConfig, build_panel, find_lessons, grade_year, make_features
from agent.stan.train import CURRICULUM, demo_history, run_training

FAST = StudyConfig(max_iter=30)


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
    exam, trades = grade_year(_year([0.9, 0.9, 0.1, 0.9], [0.10, 0.10, 0.02, -0.05]), cfg)
    assert exam["trades"] == 2 and len(trades) == 2
    assert trades[0]["ret"] == pytest.approx(1.1 * 1.1 - 1 - 0.02)
    assert trades[1]["ret"] == pytest.approx(-0.05 - 0.02)
    assert exam["win_rate"] == 0.5 and exam["passed"] is True
    expected = (1.10 - 0.01) * 1.10 * (1 - 0.01) * (1 - 0.05 - 0.01) * (1 - 0.01) - 1
    assert exam["stan_return"] == pytest.approx(expected, abs=1e-4)
    assert exam["hold_return"] == pytest.approx(1.1 * 1.1 * 1.02 * 0.95 - 1, abs=1e-4)
    assert exam["accuracy"] == 0.5  # right, right, wrong, wrong


def test_no_trades_means_no_pass():
    exam, trades = grade_year(_year([0.5, 0.5], [0.01, 0.01]), StudyConfig())
    assert trades == [] and exam["passed"] is False and exam["stan_return"] == 0.0


def test_lessons_need_enough_trades():
    few = [{"ret": -0.01, "above_ma200": False, "rsi14": 50, "vol_ratio": 1, "confidence": 0.6}] * 5
    assert find_lessons(few) == []
    many = ([{"ret": 0.02, "above_ma200": True, "rsi14": 50, "vol_ratio": 1, "confidence": 0.6}] * 40
            + [{"ret": -0.02, "above_ma200": False, "rsi14": 50, "vol_ratio": 1, "confidence": 0.6}] * 40)
    kinds = {l["kind"]: l["title"] for l in find_lessons(many)}
    assert "downtrend" in kinds["mistake"] and "uptrend" in kinds["strength"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    path = tmp_path_factory.mktemp("stan") / "stan.db"
    journal = Journal(path)
    summary = run_training(journal, demo_history(["SPY", "QQQ", "AAPL"], years=8), "practice data", demo=True, cfg=FAST)
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
    assert stan["tasks_done"] == stan["tasks_total"] == 6
    assert stan["summary"]["trades"] == summary["trades"]
    assert len(client.get("/tasks").json()) == 6
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
