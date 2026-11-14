"""How Stan studies: features, year-by-year walk-forward exams, and lessons.

Rules that keep the results honest:
- Every feature uses only data known at the close of the decision day.
- Stan is always tested on a year he did not train on, with a gap (embargo)
  between the training data and the test year.
- Trading costs are charged on every entry and exit.
- Settings are fixed up front. Nothing is tuned on the test years.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = [
    "ret1", "ret5", "ret20", "ret60", "ma20", "ma50", "ma200", "rsi14",
    "vol20", "vol_ratio", "volz", "dd252", "range", "dow",
]


@dataclass(frozen=True)
class StudyConfig:
    min_train_years: int = 3
    embargo_days: int = 5
    threshold: float = 0.55      # Stan only buys when he is at least this confident
    cost_per_side: float = 0.0005  # 0.05% each way for spread and slippage on liquid stocks
    max_iter: int = 150
    min_bucket_trades: int = 30  # a lesson needs at least this many trades behind it


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: date-indexed, sorted, columns open/high/low/close/volume for ONE symbol."""
    c = bars["close"].astype(float)
    r = c.pct_change()
    f = pd.DataFrame(index=bars.index)
    f["ret1"] = r
    f["ret5"] = c.pct_change(5)
    f["ret20"] = c.pct_change(20)
    f["ret60"] = c.pct_change(60)
    for n in (20, 50, 200):
        f[f"ma{n}"] = c / c.rolling(n).mean() - 1
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(loss > 0, 100.0)
    f.loc[gain.isna() | loss.isna(), "rsi14"] = np.nan
    f["vol20"] = r.rolling(20).std()
    f["vol_ratio"] = f["vol20"] / r.rolling(60).std()
    v = bars["volume"].astype(float)
    f["volz"] = ((v - v.rolling(20).mean()) / v.rolling(20).std()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f["dd252"] = c / c.rolling(252).max() - 1
    f["range"] = (bars["high"].astype(float) - bars["low"].astype(float)) / c
    f["dow"] = bars.index.dayofweek
    # Outcome: buy at today's close, sell at tomorrow's close.
    f["next_ret"] = c.shift(-1) / c - 1
    return f


def build_panel(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol, bars in history.items():
        bars = bars.sort_index()
        bars = bars[~bars.index.duplicated(keep="last")]
        f = make_features(bars)
        f["symbol"] = symbol
        frames.append(f)
    panel = pd.concat(frames).dropna(subset=FEATURES + ["next_ret"])
    panel.index.name = "date"
    return panel.reset_index().sort_values(["date", "symbol"]).reset_index(drop=True)


def _trades_for(group: pd.DataFrame, cost: float) -> list[dict]:
    """Turn runs of consecutive 'long' days into trades for one symbol."""
    pos = group["pos"].to_numpy()
    if not pos.any():
        return []
    run_id = (group["pos"] != group["pos"].shift()).cumsum()
    trades = []
    for _, run in group[group["pos"]].groupby(run_id[group["pos"]]):
        gross = float(np.prod(1 + run["next_ret"].to_numpy()) - 1)
        first = run.iloc[0]
        trades.append({
            "symbol": first["symbol"], "entry": first["date"], "days": len(run),
            "ret": gross - 2 * cost,
            "above_ma200": bool(first["ma200"] > 0),
            "rsi14": float(first["rsi14"]), "vol_ratio": float(first["vol_ratio"]),
            "confidence": float(first["p_up"]),
        })
    return trades


def grade_year(test: pd.DataFrame, cfg: StudyConfig) -> tuple[dict, list[dict]]:
    """Score one unseen year. Returns the exam result and the trades Stan took."""
    test = test.sort_values(["symbol", "date"]).copy()
    test["pos"] = test["p_up"] > cfg.threshold
    up = test["next_ret"] > 0
    accuracy = float(((test["p_up"] > 0.5) == up).mean())

    stan_returns, hold_returns, trades = [], [], []
    for _, g in test.groupby("symbol"):
        pos = g["pos"].astype(float)
        changes = pos.diff().abs().fillna(pos)  # entry on the first day counts
        net = pos * g["next_ret"] - cfg.cost_per_side * changes
        end_cost = cfg.cost_per_side * pos.iloc[-1]  # close anything open at year end
        stan_returns.append(float(np.prod(1 + net.to_numpy()) * (1 - end_cost) - 1))
        hold_returns.append(float(np.prod(1 + g["next_ret"].to_numpy()) - 1))
        trades.extend(_trades_for(g, cfg.cost_per_side))

    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    wins, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    profit_factor = None if not len(rets) else (99.0 if losses == 0 else round(float(wins / losses), 3))
    win_rate = float((rets > 0).mean()) if len(rets) else None
    stan_return, hold_return = float(np.mean(stan_returns)), float(np.mean(hold_returns))
    exam = {
        "year": int(test["date"].dt.year.iloc[0]),
        "n_calls": int(len(test)),
        "accuracy": round(accuracy, 4),
        "baseline_up": round(float(up.mean()), 4),
        "trades": int(len(rets)),
        "win_rate": None if win_rate is None else round(win_rate, 4),
        "profit_factor": profit_factor,
        "stan_return": round(stan_return, 4),
        "hold_return": round(hold_return, 4),
        # The bar: win at least half the trades AND make more than he loses, after costs.
        "passed": bool(win_rate is not None and win_rate >= 0.5 and (profit_factor or 0) > 1),
        "beat_hold": bool(stan_return > hold_return),
    }
    return exam, trades


def walk_forward(
    panel: pd.DataFrame,
    cfg: StudyConfig = StudyConfig(),
    on_year: Callable[[dict, list[dict], int, int], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Train on the past, sit an exam on each following year. Returns (exams, trades)."""
    years = sorted(panel["date"].dt.year.unique())
    first = panel["date"].min()
    test_years = []
    for y in years:
        cutoff = pd.Timestamp(year=y, month=1, day=1) - pd.Timedelta(days=cfg.embargo_days)
        if (cutoff - first).days >= cfg.min_train_years * 365:
            test_years.append(y)

    exams, all_trades = [], []
    for i, y in enumerate(test_years, start=1):
        cutoff = pd.Timestamp(year=y, month=1, day=1) - pd.Timedelta(days=cfg.embargo_days)
        train = panel[panel["date"] < cutoff]
        test = panel[panel["date"].dt.year == y].copy()
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=cfg.max_iter, learning_rate=0.05, random_state=7
        )
        model.fit(train[FEATURES], (train["next_ret"] > 0).astype(int))
        test["p_up"] = model.predict_proba(test[FEATURES])[:, 1]
        exam, trades = grade_year(test, cfg)
        exams.append(exam)
        all_trades.extend(trades)
        if on_year:
            on_year(exam, trades, i, len(test_years))
    return exams, all_trades


def summarize(exams: list[dict], trades: list[dict]) -> dict:
    """Overall scorecard across every unseen year."""
    n = sum(e["n_calls"] for e in exams) or 1
    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    wins, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    stan_growth = float(np.prod([1 + e["stan_return"] for e in exams]))
    hold_growth = float(np.prod([1 + e["hold_return"] for e in exams]))
    win_rate = float((rets > 0).mean()) if len(rets) else None
    pf = None if not len(rets) else (99.0 if losses == 0 else round(float(wins / losses), 3))
    ready = bool(
        exams and win_rate is not None and win_rate >= 0.5 and (pf or 0) > 1 and stan_growth > hold_growth
    )
    return {
        "years_tested": len(exams),
        "years_passed": sum(e["passed"] for e in exams),
        "years_beat_hold": sum(e["beat_hold"] for e in exams),
        "accuracy": round(sum(e["accuracy"] * e["n_calls"] for e in exams) / n, 4),
        "baseline_up": round(sum(e["baseline_up"] * e["n_calls"] for e in exams) / n, 4),
        "trades": int(len(rets)),
        "win_rate": None if win_rate is None else round(win_rate, 4),
        "profit_factor": pf,
        "stan_growth": round(stan_growth, 4),   # $1 becomes this much
        "hold_growth": round(hold_growth, 4),
        "ready_for_paper": ready,
    }


def find_lessons(trades: list[dict], cfg: StudyConfig = StudyConfig()) -> list[dict]:
    """Compare Stan's results across conditions to see where he wins and loses."""
    if not trades:
        return []
    t = pd.DataFrame(trades)
    overall = float((t["ret"] > 0).mean())
    buckets = {
        "in an uptrend (price above its 200-day average)": t["above_ma200"],
        "in a downtrend (price below its 200-day average)": ~t["above_ma200"],
        "after a sharp selloff (RSI under 30)": t["rsi14"] < 30,
        "when the stock is overbought (RSI over 70)": t["rsi14"] > 70,
        "when volatility is rising": t["vol_ratio"] > 1.2,
        "when volatility is calm": t["vol_ratio"] < 0.8,
        "on my most confident calls": t["confidence"] >= t["confidence"].quantile(0.8),
    }
    rows = []
    for label, mask in buckets.items():
        sub = t[mask]
        if len(sub) >= cfg.min_bucket_trades:
            rows.append({"when": label, "trades": int(len(sub)),
                         "win_rate": round(float((sub["ret"] > 0).mean()), 4),
                         "avg_return": round(float(sub["ret"].mean()), 5)})
    if not rows:
        return []
    rows.sort(key=lambda r: r["avg_return"])
    worst, best = rows[0], rows[-1]
    lessons = []
    if worst["avg_return"] < 0 or worst["win_rate"] < overall:
        lessons.append({"kind": "mistake", "title": f"I do worst {worst['when']}",
                        "evidence": {**worst, "overall_win_rate": round(overall, 4)}})
    if best is not worst and best["avg_return"] > 0:
        lessons.append({"kind": "strength", "title": f"I do best {best['when']}",
                        "evidence": {**best, "overall_win_rate": round(overall, 4)}})
    return lessons
