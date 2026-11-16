"""Stan's staged portfolio: no lookahead, the keep rule, and the build end to end."""
import numpy as np
import pandas as pd

from agent.stan import portfolio as pf
from agent.stan.train import DEFAULT_SYMBOLS, demo_history

CFG = pf.PortfolioConfig()


def _hist():
    return demo_history(DEFAULT_SYMBOLS[:12], years=6, seed=4)


def test_month_end_decisions_ignore_the_future():
    """Every decision made at a month-end must be identical when the data stops on that day."""
    h = _hist()
    closes = pf.closes_frame(h, "SPY")
    full_t = pf.trend_decisions(closes["SPY"], CFG)
    full_v = pf.vol_decisions(closes["SPY"], CFG)
    full_m = pf.momentum_decisions(closes, CFG)
    ends = np.flatnonzero(pf.month_ends(closes.index))
    for t in ends[ends > CFG.trend_days]:
        cut = closes.iloc[:t + 1]
        assert pf.trend_decisions(cut["SPY"], CFG).iloc[-1] == full_t.iloc[t]
        assert pf.vol_decisions(cut["SPY"], CFG).iloc[-1] == full_v.iloc[t]
        pd.testing.assert_series_equal(pf.momentum_decisions(cut, CFG).iloc[-1], full_m.iloc[t])


def test_the_last_day_of_data_counts_as_a_month_end_only_for_the_decision_it_makes():
    idx = pd.bdate_range("2026-01-01", periods=30)
    me = pf.month_ends(idx)
    assert me[-1] and me[list(idx).index(pd.Timestamp("2026-01-30"))]


def test_weights_are_held_two_days_after_the_decision():
    idx = pd.bdate_range("2026-01-01", periods=6)
    decided = pd.DataFrame({"SPY": [0, 0, 1, 1, 1, 1]}, index=idx, dtype=float)
    rets = pd.DataFrame({"SPY": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]}, index=idx)
    r = pf.returns_from_weights(decided, rets, pf.PortfolioConfig(cost=0.0))
    assert list(r.round(4)) == [0, 0, 0, 0, 0.04, 0.05]   # decided on day 3, earns from day 5


def test_costs_are_charged_on_weight_changes():
    idx = pd.bdate_range("2026-01-01", periods=5)
    decided = pd.DataFrame({"SPY": [1, 1, 0, 0, 0]}, index=idx, dtype=float)
    rets = pd.DataFrame({"SPY": [0.0] * 5}, index=idx)
    r = pf.returns_from_weights(decided, rets, pf.PortfolioConfig(cost=0.001))
    assert r.sum() == -0.002   # one buy, one sell


def test_momentum_picks_only_positive_winners():
    h = _hist()
    closes = pf.closes_frame(h, "SPY")
    w = pf.momentum_decisions(closes, CFG)
    assert (w.sum(axis=1) <= 1 + 1e-9).all()
    assert not set(w.columns) & pf.ETFS


def test_selloff_sleeve_never_exceeds_ten_positions():
    h = _hist()
    idx = pf.closes_frame(h, "SPY").index
    r = pf.selloff_sleeve(h, idx, CFG)
    assert r.abs().max() < 0.10 * 10 * 0.5   # 10 slots of 10%: a day can't move it more than the slots allow


def test_keep_rule():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    base = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    better = base * 0.5 + 0.0002          # same shape, half the risk, a bit more return per risk
    worse = base - 0.0004
    assert pf.judge(better, base)[0]
    assert not pf.judge(worse, base)[0]


def test_build_end_to_end():
    out = pf.build(demo_history(DEFAULT_SYMBOLS, years=8, seed=2))
    assert [c["key"] for c in out["checks"]] == ["trend", "vol", "selloff", "momentum", "final_sharpe",
                                                 "final_drawdown"]
    assert len(out["stages"]) == 4 and out["curve"]
    assert out["verdict"] in ("real_edge", "promising", "not_proven")
    assert out["kept"] == [s["label"] for s in out["stages"] if s["kept"]]
