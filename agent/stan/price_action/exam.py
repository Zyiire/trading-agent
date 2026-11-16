"""Stan's exam for the price-action strategy: 9 stress tests on every full 5-minute session."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Callable

import numpy as np

from agent.stan.daytrade import _p, _r
from agent.stan.price_action import risk as rk
from agent.stan.price_action.config import PriceActionConfig
from agent.stan.price_action.engine import run
from agent.stan.price_action.series import Series

BASE_COST = 0.0005   # 0.05% per side, the same as the other day-trading exams
HIGH_COST = 0.0010

RULE = ("trade only with the 15-minute structure (higher highs and lows for longs, lower for shorts); wait for a "
        "pullback to the previous day's high/low or a recent swing level, then enter after a pin bar, a "
        "high-volume engulfing candle or a failed breakout closes; stop 2 cents past the candle's wick, target 2R "
        "or the next level, stop to break-even at 1R, no entries in the first 15 minutes, flat by 3:45 PM")


def net_r(t: dict, cost: float) -> float:
    return t["R"] - 2 * cost * t["entry"] / t["risk"]


def score(trades: list[dict], cost: float = BASE_COST) -> dict:
    r = np.array([net_r(t, cost) for t in trades])
    if not len(r):
        return {"trades": 0, "win_rate": None, "profit_factor": None, "avg": None, "avg_R": None}
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": int(len(r)), "win_rate": round(float((r > 0).mean()), 4),
            "profit_factor": round(float(wins / losses), 3) if losses > 0 else 99.0, "avg": None,
            "avg_R": round(float(r.mean()), 4), "total_R": round(float(r.sum()), 2),
            "longs": sum(t["side"] == "long" for t in trades), "shorts": sum(t["side"] == "short" for t in trades),
            "affordable": round(float(np.mean([t["shares"] > 0 for t in trades])), 4)}


def _luck(trades: list[dict], series: dict[str, Series], cfg: PriceActionConfig, reps: int,
          seed: int = 29, sample: int = 600) -> dict:
    """Same stock, same entry time, same direction, same stop distance and target, on a random other
    day. If that does as well, the structure, levels and candles aren't adding anything."""
    rng = np.random.default_rng(seed)
    act = float(np.mean([net_r(t, BASE_COST) for t in trades])) if trades else None
    if len(trades) > sample:
        trades = [trades[i] for i in sorted(rng.choice(len(trades), size=sample, replace=False))]
    sims = []
    for t in trades:
        s = series.get(t["symbol"])
        if s is None or len(s.days) < 2:
            continue
        side = 1 if t["side"] == "long" else -1
        stop_pct, tgt_r = t["risk"] / t["entry"], abs(t["target"] - t["entry"]) / t["risk"]
        row = []
        for d in rng.integers(0, len(s.days), size=reps):
            idx = np.flatnonzero(s.day_idx == d)
            if len(idx) <= cfg.flatten_slot:
                continue
            k, last = idx[t["entry_bar"]], idx[cfg.flatten_slot]
            e = float(s.o[k])
            risk = e * stop_pct
            p = rk.Plan(side, e, e - side * risk, e + side * tgt_r * risk, risk, 0, "rr")
            _, px, _ = rk.walk(p, s.o, s.h, s.l, s.c, k, last, cfg)
            row.append(side * (px - e) / risk - 2 * BASE_COST / stop_pct)
        if len(row) == reps:
            sims.append(row)
    if not sims:
        return {"p_value": None, "actual_avg_R": act, "random_avg_R": None}
    rand = np.array(sims).mean(axis=0)
    p = float((np.sum(rand >= act) + 1) / (reps + 1))
    return {"p_value": round(p, 4), "actual_avg_R": round(act, 4), "random_avg_R": round(float(rand.mean()), 4),
            "reps": reps}


def price_action_exam(sessions_by_symbol: dict[str, list[tuple[str, list[dict]]]],
                      cfg: PriceActionConfig = PriceActionConfig(),
                      on_check: Callable[[dict], None] | None = None, luck_reps: int = 200) -> dict:
    cache: dict = {}
    trades = run(sessions_by_symbol, cfg, cache)
    s = score(trades)
    checks = []

    def check(key, name, passed, detail):
        item = {"key": key, "name": name, "passed": bool(passed), "detail": detail}
        checks.append(item)
        if on_check:
            on_check(item)

    days = sorted({d for v in sessions_by_symbol.values() for d, _ in v})
    check("basic", "Makes more than it loses after 0.05% costs (profit factor 1.2+, 60+ trades)",
          s["trades"] >= 60 and (s["avg_R"] or -1) > 0 and (s["profit_factor"] or 0) >= 1.2,
          f"{s['trades']} trades ({s.get('longs', 0)} long, {s.get('shorts', 0)} short): won {_p(s['win_rate'])}, "
          f"average {_r(s['avg_R'])}, profit factor {s['profit_factor']}")

    luck = _luck(trades, cache, cfg, luck_reps)
    check("luck", "Profitable, and beats the same trades placed on random days (p under 0.05)",
          luck["p_value"] is not None and luck["p_value"] < 0.05 and (luck["actual_avg_R"] or -1) > 0,
          f"its trades averaged {_r(luck['actual_avg_R'])} vs {_r(luck['random_avg_R'])} at random "
          f"(p = {luck['p_value']})")

    high = score(trades, HIGH_COST)
    check("costs", "Still ahead at 0.10% costs per side", (high["avg_R"] or -1) > 0,
          f"average {_r(high['avg_R'])} at 0.10% each way")

    longs = score([t for t in trades if t["side"] == "long"])
    shorts = score([t for t in trades if t["side"] == "short"])
    check("both_sides", "Makes money in downtrends too, not just guessing up",
          longs["trades"] >= 20 and shorts["trades"] >= 20 and (longs["avg_R"] or -1) > 0
          and (shorts["avg_R"] or -1) > 0,
          f"longs {_r(longs['avg_R'])} on {longs['trades']}, shorts {_r(shorts['avg_R'])} on {shorts['trades']}")

    by_pat = {p: score([t for t in trades if t["pattern"] == p]) for p in cfg.patterns}
    counted = {p: v for p, v in by_pat.items() if v["trades"] >= 20}
    check("patterns", "Every confirmation candle earns its keep (each with 20+ trades is ahead)",
          counted and all((v["avg_R"] or -1) > 0 for v in counted.values()),
          ", ".join(f"{p} {_r(v['avg_R'])} on {v['trades']}" for p, v in by_pat.items()))

    alt = {rr: score(run(sessions_by_symbol, replace(cfg, rr_target=rr), cache)) for rr in (1.5, 3.0)}
    check("targets", "Also ahead with a 1.5R and a 3R target, not one lucky setting",
          all((v["avg_R"] or -1) > 0 for v in alt.values()),
          f"1.5R {_r(alt[1.5]['avg_R'])}, 2R {_r(s['avg_R'])}, 3R {_r(alt[3.0]['avg_R'])}")

    weeks: dict = {}
    for t in trades:
        y, w, _ = date.fromisoformat(t["date"]).isocalendar()
        weeks.setdefault((y, w), []).append(net_r(t, BASE_COST))
    good_w = [k for k, v in weeks.items() if sum(v) > 0]
    check("weeks", "Ends at least 55% of weeks ahead",
          weeks and len(good_w) / len(weeks) >= 0.55, f"ahead in {len(good_w)} of {len(weeks)} weeks")

    per_sym: dict = {}
    for t in trades:
        per_sym.setdefault(t["symbol"], []).append(net_r(t, BASE_COST))
    enough = {k: v for k, v in per_sym.items() if len(v) >= 5}
    good = [k for k, v in enough.items() if np.mean(v) > 0]
    check("breadth", "Ahead on at least 60% of stocks", enough and len(good) / len(enough) >= 0.6,
          f"{len(good)} of {len(enough)} stocks (with 5+ trades) ahead")

    eq = np.cumsum([net_r(t, BASE_COST) for t in sorted(trades, key=lambda t: (t["date"], t["entry_bar"]))])
    dd = float((np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq).max()) if len(eq) else 0.0
    total = float(eq[-1]) if len(eq) else 0.0
    check("drawdown", "Its worst losing stretch is smaller than its total gain", total > 0 and dd < total,
          f"worst drop {dd:.1f}R, total {total:+.1f}R")

    passed = sum(c["passed"] for c in checks)
    return {
        "name": "Price action at key levels",
        "rule": RULE,
        "source": "from Zy's price-action spec: structure, levels, candles, structural stops",
        "unit": "R",
        "cost": BASE_COST,
        "stocks": len(sessions_by_symbol),
        "sessions": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "score": s,
        "by_pattern": by_pat,
        "luck": luck,
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "verdict": "real_edge" if passed == len(checks) else ("promising" if passed >= len(checks) - 2 else "not_proven"),
    }
