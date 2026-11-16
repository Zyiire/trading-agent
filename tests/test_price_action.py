"""Price-action strategy: no lookahead, each building block, and the session rules."""
import gzip
from pathlib import Path

import numpy as np
import pytest

from agent.stan import daytrade
from agent.stan.price_action import PriceActionConfig, context, run, signal_at
from agent.stan.price_action import patterns as pt
from agent.stan.price_action import risk as rk
from agent.stan.price_action import structure as st
from agent.stan.price_action.levels import Level, next_opposing
from agent.stan.price_action.series import build_htf, from_sessions

FIXTURE = Path(__file__).parent / "fixtures" / "day_trader" / "TSLA-60d5m.json.gz"
CFG = PriceActionConfig()


@pytest.fixture(scope="module")
def tsla():
    sess = daytrade.exam_sessions({"TSLA": gzip.open(FIXTURE, "rt").read()})["TSLA"]
    return from_sessions("TSLA", sess)


# ---------------------------------------------------------------- no lookahead
def test_decisions_match_when_the_future_is_cut_off(tsla):
    """For bars across the data, everything decided at bar i must be identical whether the engine
    sees the whole history or only bars 0..i. Any peek at the future would make these differ."""
    full = context(tsla, CFG)
    rng = np.random.default_rng(5)
    picks = sorted(set(rng.integers(100, len(tsla) - 1, size=60).tolist()))
    sig_bars = [i for i in range(100, len(tsla)) if signal_at(full, i)][:40]
    for i in sorted(set(picks + sig_bars)):
        cut = context(tsla.head(i + 1), CFG)
        assert cut.trend[i] == full.trend[i], f"trend differs at bar {i}"
        assert [p for p in full.pivots if p.confirmed_at <= i] == cut.pivots, f"pivots differ at bar {i}"
        assert cut.atr[i] == full.atr[i]
        assert np.isnan(cut.avg_vol[i]) and np.isnan(full.avg_vol[i]) or cut.avg_vol[i] == full.avg_vol[i]
        assert signal_at(cut, i) == signal_at(full, i), f"signal differs at bar {i}"
    assert sig_bars, "the fixture should produce some signals to check"


def test_pivots_wait_for_n_closed_htf_bars(tsla):
    htf = build_htf(tsla, CFG.htf_bars)
    for p in st.find_pivots(htf, CFG.pivot_n):
        assert p.confirmed_at == htf.close_idx[p.htf_index + CFG.pivot_n]
        assert p.confirmed_at > htf.close_idx[p.htf_index]


def test_a_still_forming_htf_bar_is_never_used(tsla):
    # cut off mid-way through a 15-minute block: that block must not exist yet
    i = next(k for k in range(200, len(tsla)) if tsla.slot[k] % CFG.htf_bars == 0)
    htf = build_htf(tsla.head(i + 1), CFG.htf_bars)
    assert htf.close_idx[-1] < i
    assert all(tsla.slot[j] % CFG.htf_bars == CFG.htf_bars - 1 or tsla.slot[j] == 77 for j in htf.close_idx)


def test_volume_average_excludes_the_signal_bar(tsla):
    ctx = context(tsla, CFG)
    i = 300
    assert ctx.avg_vol[i] == pytest.approx(tsla.v[i - CFG.vol_period:i].mean())


# ---------------------------------------------------------------- structure
def test_trend_classification():
    assert st.classify([10, 11], [8, 9], 2) == st.UP
    assert st.classify([11, 10], [9, 8], 2) == st.DOWN
    assert st.classify([10, 11], [9, 8], 2) == st.CHOP
    assert st.classify([10], [9], 2) == st.CHOP


# ---------------------------------------------------------------- patterns
def test_pin_bar():
    level = 100.0
    good = pt.Bar(o=100.30, h=100.40, l=99.80, c=100.38, v=1)       # long lower wick, closes near the high
    assert pt.pin_bar(good, 1, level, CFG) == 99.80
    no_pierce = pt.Bar(o=100.30, h=100.40, l=100.05, c=100.38, v=1)
    assert pt.pin_bar(no_pierce, 1, level, CFG) is None
    weak_close = pt.Bar(o=100.10, h=100.60, l=99.80, c=100.20, v=1)  # closes in the middle
    assert pt.pin_bar(weak_close, 1, level, CFG) is None
    short = pt.Bar(o=99.70, h=100.20, l=99.60, c=99.62, v=1)
    assert pt.pin_bar(short, -1, level, CFG) == 100.20


def test_engulfing_needs_volume():
    prev = pt.Bar(o=100.30, h=100.35, l=99.95, c=100.05, v=1000)
    b = pt.Bar(o=100.02, h=100.50, l=99.98, c=100.45, v=3000)
    assert pt.engulfing(prev, b, 1, 100.0, 0.05, avg_vol=2000, cfg=CFG) == 99.95
    assert pt.engulfing(prev, b, 1, 100.0, 0.05, avg_vol=4000, cfg=CFG) is None   # quiet volume
    small = pt.Bar(o=100.10, h=100.2, l=100.0, c=100.20, v=3000)                  # doesn't engulf
    assert pt.engulfing(prev, small, 1, 100.0, 0.05, avg_vol=2000, cfg=CFG) is None


def test_failed_breakout():
    prev = pt.Bar(o=100.1, h=100.2, l=99.7, c=99.8, v=1)   # closed below support
    b = pt.Bar(o=99.8, h=100.4, l=99.75, c=100.3, v=1)     # and straight back above
    assert pt.failed_breakout(prev, b, 1, 100.0) == 99.7
    held = pt.Bar(o=99.8, h=99.95, l=99.6, c=99.7, v=1)    # breakdown held: no trade
    assert pt.failed_breakout(prev, held, 1, 100.0) is None


# ---------------------------------------------------------------- risk
def test_stop_size_and_target():
    cfg = PriceActionConfig(account=1000, risk_dollars=10)
    p = rk.plan(1, entry=100.0, structure=99.5, swing=None, opposing=None, cfg=cfg)
    assert p.stop == 99.48 and p.risk == pytest.approx(0.52)
    assert p.shares == 10                     # floor(10 / 0.52) = 19, but $1000 buys only 10
    assert p.target == pytest.approx(101.04) and p.target_kind == "rr"
    near = rk.plan(1, 100.0, 99.5, None, opposing=100.8, cfg=cfg)
    assert near.target == 100.8 and near.target_kind == "level"   # the level comes first
    assert rk.plan(1, 100.0, 99.5, None, opposing=100.3, cfg=cfg) is None  # less than 1R of room


def test_size_uses_percent_risk_when_no_dollar_amount():
    cfg = PriceActionConfig(account=10000, risk_dollars=None, risk_pct=0.01)
    assert rk.size(50.0, 0.5, cfg) == 200


def test_walk_breakeven_and_stop_first():
    cfg = PriceActionConfig()
    p = rk.Plan(1, 100.0, 99.0, 102.0, 1.0, 1, "rr")
    o = np.array([100.0, 100.5, 100.4]); h = np.array([100.2, 101.1, 100.5])
    l = np.array([99.8, 100.3, 99.9]); c = np.array([100.1, 100.6, 100.0])
    assert rk.walk(p, o, h, l, c, 0, 2, cfg) == (2, 100.0, "breakeven")   # +1R reached, then back to entry
    both = rk.walk(p, np.array([100.0]), np.array([102.5]), np.array([98.9]), np.array([101.0]), 0, 0, cfg)
    assert both[2] == "stop"                                              # one bar hit both: assume the stop


def test_next_opposing_level():
    lv = [Level("PDH", 103.0), Level("swing high", 101.5), Level("PDL", 98.0)]
    assert next_opposing(lv, 100.0, 1, 0.1) == 101.5
    assert next_opposing(lv, 100.0, -1, 0.1) == 98.0


# ---------------------------------------------------------------- the whole engine
def test_session_rules_on_real_bars(tsla):
    sess = daytrade.exam_sessions({"TSLA": gzip.open(FIXTURE, "rt").read()})
    trades = run(sess)
    assert trades
    per_day = {}
    for t in trades:
        assert t["signal_slot"] >= CFG.first_entry_slot          # nothing from the first 15 minutes
        assert t["exit_bar"] <= CFG.flatten_slot                 # flat by 3:45 PM
        assert t["entry_bar"] == t["signal_slot"] + 1            # filled at the next bar's open
        assert t["R"] >= -1.5 or t["outcome"] == "stop"          # only a gap through the stop loses more
        per_day[t["date"]] = per_day.get(t["date"], 0) + 1
    assert max(per_day.values()) <= CFG.max_trades_per_day


def test_trend_direction_decides_the_side(tsla):
    ctx = context(tsla, CFG)
    for i in range(1, len(tsla)):
        sig = signal_at(ctx, i)
        if sig:
            assert sig.side == ctx.trend[i] != st.CHOP


def test_config_validation():
    with pytest.raises(ValueError):
        PriceActionConfig(htf_minutes=12)
    with pytest.raises(ValueError):
        PriceActionConfig(fill="whenever")


def test_exam_finds_nothing_on_random_walk_demo_data():
    from agent.stan.price_action.exam import price_action_exam
    sess = daytrade.exam_sessions(daytrade.load_charts(["SPY", "QQQ", "AAPL", "MSFT", "AMD"], demo=True))
    exam = price_action_exam(sess, luck_reps=60)
    assert exam["checks_total"] == 9 and exam["verdict"] == "not_proven"
    assert exam["score"]["avg_R"] < 0
