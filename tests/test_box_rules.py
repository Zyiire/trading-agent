"""Box-breakout rules on synthetic 1-minute bars (no network, no database)."""
from datetime import time

import numpy as np
import pandas as pd
import pytest

from agent.stan.box import rules

ET = rules.ET


def day_minutes(day="2024-03-04", base=100.0, edits=None, drop=()):
    """One day of flat 1-minute bars 9:00-15:59 at `base`. `edits` maps "HH:MM" to (o, h, l, c)."""
    idx = pd.date_range(f"{day} 09:00", f"{day} 15:59", freq="1min", tz=ET)
    df = pd.DataFrame({k: base for k in "ohlc"}, index=idx)
    for hhmm, (o, h, l, c) in (edits or {}).items():
        df.loc[pd.Timestamp(f"{day} {hhmm}", tz=ET)] = [o, h, l, c]
    gone = [pd.Timestamp(f"{day} {t}", tz=ET) for t in drop]
    return df.drop(index=gone)


def boxed(**kw):
    """Box high 101, low 99 (set on the 9:00 minute)."""
    edits = {"09:00": (100, 101, 99, 100)}
    edits.update(kw.pop("edits", {}))
    return day_minutes(edits=edits, **kw)


def five(df):
    return rules.bars(df, 5)


def test_bars_builds_ohlc_and_accepts_rows():
    df = day_minutes(edits={"09:01": (100, 105, 95, 101), "09:04": (100, 100, 100, 107)})
    b = rules.bars(df, 5).loc[pd.Timestamp("2024-03-04 09:00", tz=ET)]
    assert (b.o, b.h, b.l, b.c) == (100, 105, 95, 107)
    rows = [(int(t.timestamp()), r.o, r.h, r.l, r.c, 1000, 0.0) for t, r in df.iterrows()]
    assert rules.bars(rows, 5).equals(rules.bars(df, 5))
    assert rules.bars([], 5).empty


def test_box_from_six_bars():
    assert rules.box(five(boxed())) == (101.0, 99.0)


def test_box_ignores_bars_outside_nine_to_nine_thirty():
    df = boxed(edits={"09:30": (100, 150, 50, 100)})
    assert rules.box(five(df)) == (101.0, 99.0)


def test_box_missing_bar_is_none():
    df = boxed(drop=[f"09:{m:02d}" for m in range(10, 15)])
    assert rules.box(five(df)) is None
    assert rules.box(five(df.iloc[0:0])) is None


def test_long_breakout_enters_at_close_stop_opposite_target_3r():
    df = boxed(edits={"09:34": (100, 102, 100, 102)})
    sig = rules.signal(five(df), (101, 99))
    assert sig.side == "long" and sig.entry == 102 and sig.stop == 99
    assert sig.risk == 3 and sig.target == 111
    assert sig.bar_open == pd.Timestamp("2024-03-04 09:30", tz=ET)


def test_short_breakout():
    df = boxed(edits={"09:44": (100, 100, 98, 98)})
    sig = rules.signal(five(df), (101, 99))
    assert sig.side == "short" and sig.entry == 98 and sig.stop == 101
    assert sig.risk == 3 and sig.target == 89
    assert sig.bar_open == pd.Timestamp("2024-03-04 09:40", tz=ET)


def test_first_close_outside_wins():
    df = boxed(edits={"09:39": (100, 102, 100, 102), "09:44": (100, 100, 97, 97)})
    assert rules.signal(five(df), (101, 99)).side == "long"


def test_wick_through_box_without_close_is_no_signal():
    df = boxed(edits={"09:32": (100, 110, 90, 100)})
    assert rules.signal(five(df), (101, 99)) is None


def test_no_breakout_by_ten_thirty_is_none():
    df = boxed(edits={"10:34": (100, 102, 100, 102)})   # the 10:30 bar closes after the window
    assert rules.signal(five(df), (101, 99)) is None


def test_last_signal_bar_opens_at_ten_twenty_five():
    df = boxed(edits={"10:29": (100, 102, 100, 102)})
    assert rules.signal(five(df), (101, 99)).bar_open == pd.Timestamp("2024-03-04 10:25", tz=ET)


def test_breakout_before_open_does_not_count():
    df = boxed(edits={"09:29": (100, 103, 100, 103)})   # closes the 9:25 bar, which is in the box
    five_day = five(df)
    assert rules.box(five_day) == (103.0, 99.0)
    assert rules.signal(five_day, rules.box(five_day)) is None


def test_trend_blocks_counter_trend_and_unknown():
    df = boxed(edits={"09:34": (100, 102, 100, 102)})
    f = five(df)
    assert rules.signal(f, (101, 99), trend=lambda at: True).side == "long"
    assert rules.signal(f, (101, 99), trend=lambda at: False) is None
    assert rules.signal(f, (101, 99), trend=lambda at: None) is None


def test_trend_receives_breakout_bar_close_time():
    seen = []
    df = boxed(edits={"09:34": (100, 102, 100, 102)})
    rules.signal(five(df), (101, 99), trend=lambda at: seen.append(at) or True)
    assert seen == [pd.Timestamp("2024-03-04 09:35", tz=ET)]


def fifteen_series(up=300, down=300):
    idx = pd.date_range("2024-01-02 04:00", periods=up + down, freq="15min", tz=ET)
    close = np.concatenate([100 + np.arange(up) * 1.0, 100 + up - np.arange(1, down + 1) * 2.0])
    return pd.DataFrame({"o": close, "h": close, "l": close, "c": close}, index=idx)


def test_trend_up_direction():
    f = fifteen_series()
    assert rules.trend_up(f, f.index[200] + pd.Timedelta(minutes=15)) is True
    assert rules.trend_up(f, f.index[-1] + pd.Timedelta(minutes=15)) is False
    assert rules.trend_up(f, f.index[0]) is None      # nothing has finished yet


def test_trend_up_uses_only_finished_bars():
    f = fifteen_series()
    flags = rules.trend_flags(f)
    i = int(np.argmax(~flags.to_numpy() & (np.arange(len(f)) > 300)))   # first bar after the peak that flips
    assert flags.iloc[i - 1] and not flags.iloc[i]
    bar = f.index[i]
    assert rules.trend_up(f, bar + pd.Timedelta(minutes=14)) is True     # bar i still forming
    assert rules.trend_up(f, bar + pd.Timedelta(minutes=15)) is False    # bar i has finished
    assert rules.trend_up(f, bar + pd.Timedelta(minutes=14), flags) is True
    assert rules.trend_up(f, bar + pd.Timedelta(minutes=15), flags) is False


def long_sig():
    return rules.Signal("long", pd.Timestamp("2024-03-04 09:30", tz=ET), 102.0, 99.0, 111.0, 3.0)


def after(df, hhmm="09:35"):
    return df[df.index >= pd.Timestamp(f"2024-03-04 {hhmm}", tz=ET)]


def test_exit_stop_target_time():
    sig = long_sig()
    assert rules.simulate_exit(after(day_minutes(edits={"10:00": (100, 100, 98, 100)})), sig) == (99.0, "stop")
    assert rules.simulate_exit(after(day_minutes(edits={"10:00": (100, 112, 100, 100)})), sig) == (111.0, "target")
    px, why = rules.simulate_exit(after(day_minutes(edits={"15:55": (100, 100, 100, 103.5)})), sig)
    assert (px, why) == (103.5, "time")


def test_exit_same_bar_touching_both_is_the_stop():
    df = day_minutes(edits={"10:00": (100, 112, 98, 100)})
    assert rules.simulate_exit(after(df), long_sig()) == (99.0, "stop")


def test_exit_target_before_stop_on_earlier_bar():
    df = day_minutes(edits={"10:00": (100, 112, 100, 100), "10:01": (100, 100, 90, 100)})
    assert rules.simulate_exit(after(df), long_sig()) == (111.0, "target")


def test_exit_short_side():
    sig = rules.Signal("short", pd.Timestamp("2024-03-04 09:30", tz=ET), 98.0, 101.0, 89.0, 3.0)
    assert rules.simulate_exit(after(day_minutes(edits={"10:00": (100, 102, 100, 100)})), sig) == (101.0, "stop")
    assert rules.simulate_exit(after(day_minutes(edits={"10:00": (100, 100, 88, 100)})), sig) == (89.0, "target")
    assert rules.simulate_exit(after(day_minutes(edits={"10:00": (100, 102, 88, 100)})), sig) == (101.0, "stop")


def test_time_exit_is_the_355_bar_and_ignores_later_minutes():
    df = day_minutes(edits={"15:55": (100, 100, 100, 104), "15:56": (100, 200, 0, 150)})
    assert rules.simulate_exit(after(df), long_sig()) == (104.0, "time")
    assert rules.simulate_exit(after(df), long_sig(), flat=time(15, 54)) == (100.0, "time")


def test_exit_without_bars_is_none():
    assert rules.simulate_exit(after(day_minutes(), "16:30"), long_sig()) is None


def test_size_one_percent_rule():
    assert rules.size(100_000, 2.0, 400_000, 50.0) == 500
    assert rules.size(100_000, 3.0, 400_000, 50.0) == 333          # floors
    assert rules.size(100_000, 2.0, 400_000, 50.0, risk_pct=0.005) == 250


def test_size_under_one_share_is_zero():
    assert rules.size(1_000, 50.0, 4_000, 100.0) == 0
    assert rules.size(0, 1.0, 1_000, 10.0) == 0
    assert rules.size(10_000, 0.0, 1_000, 10.0) == 0


def test_size_down_to_95_percent_of_buying_power():
    assert rules.size(100_000, 1.0, 100_000, 100.0) == 950      # 1,000 wanted; $95k buys 950 (risk 0.95%)
    assert rules.size(100_000, 1.0, 200_000, 100.0) == 1000     # $100k fits
    assert rules.size(100_000, 1.0, 105_300, 100.0) == 1000     # $100k is under 95% of 105,300
    assert rules.size(100_000, 1.0, 105_200, 100.0) == 999      # just over: one share fewer
    assert rules.size(100_000, 1.0, 100.0, 100.0) == 0          # not even one share fits


def test_size_requires_price():
    with pytest.raises(TypeError):
        rules.size(100_000, 1.0, 200_000)


def test_bars_now_drops_unfinished_buckets():
    df = boxed(edits={"09:34": (100, 102, 100, 102)})
    at = lambda hhmm, sec=0: pd.Timestamp(f"2024-03-04 {hhmm}", tz=ET) + pd.Timedelta(seconds=sec)
    assert at("09:25") in rules.bars(df, 5, now=at("09:30")).index          # finished exactly at 9:30:00
    assert at("09:25") not in rules.bars(df, 5, now=at("09:29", 59)).index  # still forming
    live = rules.bars(df[df.index < at("09:33")], 5, now=at("09:33"))       # 9:30 bar half built at 9:33
    assert at("09:30") not in live.index
    assert rules.signal(live, (101, 99)) is None
    full = rules.bars(df, 5, now=at("09:35"))
    assert rules.signal(full, (101, 99)).side == "long"


def test_box_ignores_half_built_last_bar_when_now_given():
    df = boxed(edits={"09:24": (100, 103, 100, 100)})      # 9:25 bar would add a 103 high
    early = rules.bars(df[df.index < pd.Timestamp("2024-03-04 09:23", tz=ET)], 5,
                       now=pd.Timestamp("2024-03-04 09:23", tz=ET))
    assert rules.box(early) is None
