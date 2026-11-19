"""Stan's box-breakout report card: the video's NQ rules, replayed on QQQ 1-minute bars.

box_exam(minutes) takes an ET-indexed o/h/l/c minute DataFrame (see data.load) and returns a plain
dict. Every trade comes from the same rules.py functions the live bot uses.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import time

import numpy as np
import pandas as pd

from agent.stan.box import rules

POINT = 1 / 41.0          # one NQ point, in QQQ dollars (NQ is about 41x QQQ)
SPLIT_YEAR = 2023         # in-sample is before, out-of-sample from here
MC_RUNS, MC_SEED = 1000, 7
RISK_PCT = 0.01
TARGETS = (2.5, 3.0, 3.5, 4.0)
FLATS = (time(15, 55), time(15, 59))

# What the video claimed on NQ.
VIDEO = {
    "no_filter": {"trades": 2727, "avg_r": 0.08, "max_dd_r": -41.0},
    "headline": {"trades": 1371, "win_rate": 0.373, "avg_r": 0.17, "pf": "1.23-1.26", "max_dd_r": -26.0,
                 "total_r": 232.0},
    "positive_years": "9 of 11",
    "split": {"early_avg_r": 0.15, "late_avg_r": 0.21},
    "monte_carlo": {"median_dd": 26.7, "p95_dd": 43.8},
}


def _drawdown(r: np.ndarray) -> float:
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]).min())


def stats(t: pd.DataFrame) -> dict:
    """Headline numbers for a trades frame (column r, in R)."""
    if t.empty:
        return {"trades": 0, "win_rate": None, "avg_r": None, "pf": None, "total_r": 0.0,
                "max_dd_r": 0.0, "lose_streak": 0}
    r = t["r"].to_numpy()
    streak = longest = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        longest = max(longest, streak)
    losses = -r[r <= 0].sum()
    return {"trades": int(len(r)), "win_rate": float((r > 0).mean()), "avg_r": float(r.mean()),
            "pf": float(r[r > 0].sum() / losses) if losses else None, "total_r": float(r.sum()),
            "max_dd_r": _drawdown(r), "lose_streak": int(longest)}


def _candidates(minutes: pd.DataFrame, trend: bool) -> list[tuple]:
    """(day, signal, box height, minute bars from the entry on) for every day that has a signal."""
    five = rules.bars(minutes, 5)
    fifteen = rules.bars(minutes, 15)
    flags = rules.trend_flags(fifteen)
    follow = (lambda at: rules.trend_up(fifteen, at, flags)) if trend else None
    min_days = {d: g for d, g in minutes.groupby(minutes.index.date)}
    out = []
    for day, bars5 in five.groupby(five.index.date):
        bx = rules.box(bars5)
        if bx is None or day not in min_days:
            continue
        sig = rules.signal(bars5, bx, follow)
        if sig is None:
            continue
        today = min_days[day]
        after = today.iloc[today.index.searchsorted(sig.bar_open + pd.Timedelta(minutes=5)):]
        out.append((day, sig, bx[0] - bx[1], after))
    return out


def _trades(cands: list[tuple], target_r: float = rules.TARGET_R, flat: time = rules.FLAT,
            cost_points: float = 1.0) -> pd.DataFrame:
    rows = []
    for day, sig, height, after in cands:
        if target_r != rules.TARGET_R:
            sign = 1 if sig.side == "long" else -1
            sig = replace(sig, target=sig.entry + sign * target_r * sig.risk)
        res = rules.simulate_exit(after, sig, flat)
        if res is None:
            continue
        px, why = res
        pnl = (px - sig.entry) if sig.side == "long" else (sig.entry - px)
        rows.append({"day": pd.Timestamp(day), "side": sig.side, "r": (pnl - cost_points * POINT) / sig.risk,
                     "why": why, "risk": sig.risk, "box": height})
    t = pd.DataFrame(rows, columns=["day", "side", "r", "why", "risk", "box"])
    t["day"] = pd.to_datetime(t["day"])
    return t


def _monte_carlo(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"runs": MC_RUNS, "median_dd": None, "p95_dd": None}
    rng = np.random.default_rng(MC_SEED)
    dds = [-_drawdown(rng.permutation(r)) for _ in range(MC_RUNS)]
    return {"runs": MC_RUNS, "median_dd": float(np.median(dds)), "p95_dd": float(np.percentile(dds, 95))}


def _worst_drop(curve: np.ndarray) -> float:
    return float((curve / np.maximum.accumulate(curve) - 1).min())


def _money(t: pd.DataFrame, minutes: pd.DataFrame) -> dict:
    daily = minutes["c"].groupby(minutes.index.date).last().to_numpy()
    out = {"risk_pct": RISK_PCT, "growth": None, "worst_drop": None, "years": None,
           "hold_growth": float(daily[-1] / daily[0]), "hold_worst_drop": _worst_drop(daily)}
    if not t.empty:
        curve = np.cumprod(1 + RISK_PCT * t["r"].to_numpy())
        out.update(growth=float(curve[-1]), worst_drop=_worst_drop(np.concatenate([[1.0], curve])),
                   years=float((t["day"].iloc[-1] - t["day"].iloc[0]).days / 365.25))
    return out


def _by_year(t: pd.DataFrame) -> list[dict]:
    return [{"year": int(y), **stats(g)} for y, g in t.groupby(t["day"].dt.year)]


def _facts(res: dict, span: tuple[int, int]) -> list[str]:
    h, nf, c2, split, mc, money, v = (res[k] for k in
                                      ("headline", "no_filter", "cost_2x", "split", "monte_carlo", "money", "video"))
    if not h["trades"]:
        return [f"The box breakout found no trades on QQQ between {span[0]} and {span[1]}."]
    yrs = res["by_year"]
    pos = sum(1 for y in yrs if (y["avg_r"] or 0) > 0)
    neg = [str(y["year"]) for y in yrs if (y["avg_r"] or 0) <= 0]
    pf = f"{h['pf']:.2f}" if h["pf"] is not None else "n/a"
    facts = [
        f"I replayed the video's box breakout on QQQ 1-minute bars from {span[0]} to {span[1]}, "
        f"with a cost of 1 NQ point per trade.",
        f"With the trend filter it took {h['trades']:,} trades, won {h['win_rate'] * 100:.1f}% of them, "
        f"and averaged {h['avg_r']:+.3f}R a trade with a profit factor of {pf}.",
        f"The total was {h['total_r']:+.1f}R, and the worst drop was {abs(h['max_dd_r']):.1f}R.",
        f"Without the trend filter it took {nf['trades']:,} trades and averaged {nf['avg_r']:+.3f}R, "
        f"so the filter helped." if (nf["avg_r"] or 0) < h["avg_r"] else
        f"Without the trend filter it took {nf['trades']:,} trades and averaged {nf['avg_r']:+.3f}R, "
        f"so the filter did not help.",
        f"{pos} of {len(yrs)} years were positive" + (f"; the losing years were {', '.join(neg)}." if neg else "."),
    ]
    e, l = split["early"], split["late"]
    if e["trades"] and l["trades"]:
        facts.append(f"Before {SPLIT_YEAR} it averaged {e['avg_r']:+.3f}R a trade. "
                     f"From {SPLIT_YEAR} on it averaged {l['avg_r']:+.3f}R.")
    if c2["avg_r"] is not None:
        facts.append(f"If costs double to 2 NQ points, the average falls to {c2['avg_r']:+.3f}R a trade.")
    if mc["median_dd"] is not None:
        facts.append(f"Shuffling the trades 1,000 times, the typical worst drop was {mc['median_dd']:.1f}R "
                     f"and 1 time in 20 it was {mc['p95_dd']:.1f}R or worse.")
    if money["growth"] is not None:
        facts.append(f"Risking 1% of the account per trade, $1 became ${money['growth']:.2f} with a worst drop of "
                     f"{abs(money['worst_drop']) * 100:.1f}%. Holding QQQ turned $1 into ${money['hold_growth']:.2f} "
                     f"with a worst drop of {abs(money['hold_worst_drop']) * 100:.1f}%.")
    vh = v["headline"]
    facts.append(f"The video claimed {vh['trades']:,} trades on NQ, {vh['win_rate'] * 100:.1f}% wins and "
                 f"{vh['avg_r']:+.2f}R a trade, a profit factor of {vh['pf']} and a worst drop of "
                 f"{abs(vh['max_dd_r']):.0f}R. I got {h['trades']:,} trades, {h['win_rate'] * 100:.1f}% wins and "
                 f"{h['avg_r']:+.3f}R, a profit factor of {pf} and a worst drop of {abs(h['max_dd_r']):.1f}R.")
    if h["avg_r"] <= 0:
        facts.append("So the video's edge did not show up on QQQ in this data.")
    elif h["avg_r"] < vh["avg_r"]:
        facts.append("So the rules still paid on QQQ, but less per trade than the video claimed.")
    else:
        facts.append("So the rules paid on QQQ at least as well as the video claimed.")
    return facts


def box_exam(minutes: pd.DataFrame, cost_points: float = 1.0) -> dict:
    """Replay the rules on every day in `minutes` and build the report card."""
    minutes = rules.minute_frame(minutes)
    if minutes.empty:
        return {"headline": stats(pd.DataFrame(columns=["r"])), "no_filter": stats(pd.DataFrame(columns=["r"])),
                "cost_2x": stats(pd.DataFrame(columns=["r"])), "by_year": [], "split": {}, "grid": [],
                "monte_carlo": _monte_carlo(np.array([])), "money": {}, "video": VIDEO,
                "facts": ["There were no minute bars to study."], "trades": 0}
    trend_c, plain_c = _candidates(minutes, True), _candidates(minutes, False)
    main = _trades(trend_c, cost_points=cost_points)
    heavy = _trades(trend_c, cost_points=2 * cost_points)
    days = main["day"] if not main.empty else pd.Series(minutes.index)
    first, last = int(days.dt.year.min()), int(days.dt.year.max())
    early, late = main[main["day"].dt.year < SPLIT_YEAR], main[main["day"].dt.year >= SPLIT_YEAR]
    res = {
        "cost_points": cost_points,
        "span": [first, last],
        "headline": stats(main),
        "no_filter": stats(_trades(plain_c, cost_points=cost_points)),
        "cost_2x": {**stats(heavy), "late_avg_r": stats(heavy[heavy["day"].dt.year >= SPLIT_YEAR])["avg_r"]},
        "by_year": _by_year(main),
        "split": {"year": SPLIT_YEAR, "early": stats(early), "late": stats(late)},
        "grid": [{"target": tr, "flat": f"{fl.hour}:{fl.minute:02d}", **{
                      k: stats(_trades(trend_c, tr, fl, cost_points))[k] for k in ("avg_r", "total_r")}}
                 for tr in TARGETS for fl in FLATS],
        "exits": {k: int(v) for k, v in main["why"].value_counts().items()},
        "monte_carlo": _monte_carlo(main["r"].to_numpy()),
        "money": _money(main, minutes),
        "video": VIDEO,
    }
    res["facts"] = _facts(res, (first, last))
    return res
