"""The box exam on a small synthetic set with known outcomes."""
import pandas as pd
import pytest

from agent.stan.box import exam, rules
from tests.test_box_rules import day_minutes

COST = exam.POINT
KEYS = {"headline", "no_filter", "cost_2x", "by_year", "split", "grid", "monte_carlo", "money", "video", "facts"}


def synthetic(n_days=32, warmup=2, missing_box_day=None):
    """A rising market. After two quiet warm-up days, days cycle through five patterns:
    0 long to target, 1 long to stop, 2 short (against the trend), 3 no breakout, 4 long to the 3:55 exit."""
    days = pd.bdate_range("2024-01-02", periods=n_days)
    frames = []
    for i, d in enumerate(days):
        base = 100.0 + i
        k = (i - warmup) % 5 if i >= warmup else 3
        e = {"09:00": (base, base + 1, base - 1, base)}
        if k in (0, 1, 4):
            e["09:34"] = (base, base + 2, base, base + 2)
        if k == 2:
            e["09:34"] = (base, base, base - 2, base - 2)
        if k == 0:
            e["10:00"] = (base, base + 12, base, base)
        if k == 1:
            e["10:00"] = (base, base, base - 2, base)
        drop = [f"09:{m:02d}" for m in range(10, 15)] if i == missing_box_day else ()
        frames.append(day_minutes(day=d.strftime("%Y-%m-%d"), base=base, edits=e, drop=drop))
    return pd.concat(frames)


def test_exam_has_every_key_and_counts():
    res = exam.box_exam(synthetic())
    assert KEYS <= set(res)
    # days 2..31 = 30 days, six of each pattern; trend allows longs only
    assert res["headline"]["trades"] == 18
    assert res["no_filter"]["trades"] == 24
    assert res["exits"] == {"target": 6, "stop": 6, "time": 6}
    assert res["headline"]["win_rate"] == pytest.approx(6 / 18)


def test_exam_r_values_include_cost():
    res = exam.box_exam(synthetic())
    target_r, stop_r, time_r = (9 - COST) / 3, (-3 - COST) / 3, (-2 - COST) / 3
    assert res["headline"]["total_r"] == pytest.approx(6 * (target_r + stop_r + time_r))
    heavy = 6 * ((9 - 2 * COST) / 3 + (-3 - 2 * COST) / 3 + (-2 - 2 * COST) / 3)
    assert res["cost_2x"]["total_r"] == pytest.approx(heavy)
    assert res["headline"]["avg_r"] == pytest.approx((target_r + stop_r + time_r) / 3)


def test_exam_missing_box_day_is_skipped():
    full = exam.box_exam(synthetic())["headline"]["trades"]
    # day 2 is a pattern-0 long day; remove part of its box
    assert exam.box_exam(synthetic(missing_box_day=2))["headline"]["trades"] == full - 1


def test_exam_split_years_grid_and_money():
    res = exam.box_exam(synthetic())
    assert res["by_year"][0]["year"] == 2024 and res["by_year"][0]["trades"] == 18
    assert res["split"]["early"]["trades"] == 0 and res["split"]["late"]["trades"] == 18
    assert len(res["grid"]) == len(exam.TARGETS) * len(exam.FLATS)
    g = {(x["target"], x["flat"]): x for x in res["grid"]}
    assert g[(3.0, "15:55")]["total_r"] == pytest.approx(res["headline"]["total_r"])
    assert res["monte_carlo"]["runs"] == 1000 and res["monte_carlo"]["median_dd"] > 0
    assert res["money"]["risk_pct"] == 0.01 and 1 < res["money"]["growth"] and res["money"]["hold_growth"] > 1
    assert res["video"]["headline"]["trades"] == 1371


def test_exam_facts_are_plain_sentences_with_numbers():
    facts = exam.box_exam(synthetic())["facts"]
    assert all(isinstance(f, str) and f.endswith(".") and "{" not in f and "nan" not in f.lower() for f in facts)
    text = " ".join(facts)
    assert "18 trades" in text and "1,371" in text and "video" in text.lower()


def test_exam_is_json_serialisable():
    import json
    json.dumps(exam.box_exam(synthetic()))


def test_exam_with_no_trades_does_not_crash():
    res = exam.box_exam(day_minutes())          # no box breakout at all
    assert KEYS <= set(res) and res["headline"]["trades"] == 0
    assert res["facts"] and exam.box_exam(day_minutes().iloc[0:0])["facts"]


def _train(journal):
    from agent.stan.train import run_training
    from tests.test_stan import FAST
    from agent.stan.train import demo_history
    run_training(journal, demo_history(["SPY", "QQQ"], years=6), "test data", demo=False, cfg=FAST)


def test_training_runs_box_exam_and_saves_meta(tmp_path, monkeypatch):
    import sqlite3
    from agent.stan import journal as jmod
    from agent.stan.box import data
    df = synthetic()
    path = tmp_path / "box.db"
    with sqlite3.connect(path) as db:
        db.execute(data.SCHEMA)
        db.executemany("INSERT INTO bars_1m VALUES ('QQQ',?,?,?,?,?,0,0,'t')",
                       [(int(t.timestamp()), r.o, r.h, r.l, r.c) for t, r in df.iterrows()])
    monkeypatch.setattr(data, "DB_PATH", path)
    j = jmod.Journal(":memory:")
    _train(j)
    assert {t["key"]: t["status"] for t in j.tasks()}["box"] == "done"
    assert j.meta()["box_exam"]["headline"]["trades"] == 18
    assert any("1,371" in t["text"] for t in j.thoughts(limit=1000))


def test_training_skips_box_exam_when_db_missing(tmp_path, monkeypatch):
    from agent.stan import journal as jmod
    from agent.stan.box import data
    monkeypatch.setattr(data, "DB_PATH", tmp_path / "nope.db")
    j = jmod.Journal(":memory:")
    _train(j)
    assert {t["key"]: t["status"] for t in j.tasks()}["box"] == "failed"
    assert "box_exam" not in j.meta() and j.meta()["state"] == "finished"


def test_live_risk_is_capped_by_buying_power():
    t = pd.DataFrame({"risk": [2.0, 0.2, 0.05], "entry": [100.0, 100.0, 100.0], "r": [1.0, 1.0, -1.0]})
    # cap = 0.95 x 4 x risk / entry: 7.6% (1% applies), 0.76%, 0.19%
    assert list(exam.live_risk(t)) == pytest.approx([0.01, 0.0076, 0.0019])


def test_exam_live_money_view_and_facts():
    res = exam.box_exam(synthetic())
    live = res["live_money"]
    assert live["leverage"] == 4.0 and live["growth"] is not None and live["worst_drop"] <= 0
    trades = res["headline"]["trades"]
    assert 0 <= live["sized_down"] <= trades and live["sized_down_share"] == pytest.approx(live["sized_down"] / trades)
    assert live["growth"] <= res["money"]["growth"] + 1e-12 or live["sized_down"] == 0
    text = " ".join(res["facts"])
    assert "On narrow-box days the position is capped by buying power, so about" in text
    assert "trades risk less than 1%." in text and "With live sizing" in text


def test_exam_facts_are_self_contained():
    res = exam.box_exam(synthetic())
    res["by_year"] = [{"year": 2016, "avg_r": 0.1}, {"year": 2017, "avg_r": -0.1}, {"year": 2020, "avg_r": -0.02},
                      {"year": 2021, "avg_r": 0.2}]
    f = exam.facts(res)
    assert "The box breakout lost money in 2017 and 2020; every other year was positive (2 of 4 years from 2024 to 2024)." in f
    assert any(x.startswith("Versus simply holding QQQ:") for x in f)
    assert any(x.startswith("If costs (fees and slippage) double") and "box breakout" in x for x in f)
    assert all("box breakout" in x.lower() or x.startswith("The video claimed") for x in f)


def test_facts_from_an_older_saved_exam_without_live_money():
    res = exam.box_exam(synthetic())
    res.pop("live_money")
    assert not any("live sizing" in x for x in exam.facts(res))
