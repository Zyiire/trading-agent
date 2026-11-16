"""How Stan studies: features, year-by-year walk-forward exams, the report card,
a luck test, and lessons.

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
    lesson_gap: float = 0.03     # a condition must differ from average by 3 points to count
    lesson_margin: float = 0.0005  # ...and by 0.05% per trade on average
    luck_reps: int = 1000


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    out = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(loss > 0, 100.0)
    out[gain.isna() | loss.isna()] = np.nan
    return out


def max_drawdown(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    curve = (1 + daily_returns).cumprod()
    return float((curve / curve.cummax() - 1).min())


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
    f["rsi14"] = rsi(c)
    f["vol20"] = r.rolling(20).std()
    f["vol_ratio"] = f["vol20"] / r.rolling(60).std()
    v = bars["volume"].astype(float)
    f["volz"] = ((v - v.rolling(20).mean()) / v.rolling(20).std()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f["dd252"] = c / c.rolling(252).max() - 1
    f["range"] = (bars["high"].astype(float) - bars["low"].astype(float)) / c
    f["dow"] = bars.index.dayofweek
    f["close"] = c
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
    if not group["pos"].any():
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


def daily_portfolio(test: pd.DataFrame, cost: float) -> tuple[pd.Series, pd.Series, float]:
    """Equal money in every stock. Returns (Stan's daily returns, holding's daily returns, time invested)."""
    t = test.sort_values(["symbol", "date"]).copy()
    pos = t["pos"].astype(float)
    change = pos.groupby(t["symbol"]).diff().abs().fillna(pos)
    last_day = t.groupby("symbol")["date"].transform("max") == t["date"]
    change = change + np.where(last_day, pos, 0.0)  # close anything still open at year end
    t["net"] = pos * t["next_ret"] - cost * change
    stan = t.groupby("date")["net"].mean()
    hold = t.groupby("date")["next_ret"].mean()
    return stan, hold, float(pos.mean())


def grade_year(test: pd.DataFrame, cfg: StudyConfig) -> tuple[dict, list[dict], dict]:
    """Score one unseen year. Returns (exam result, trades, daily return series)."""
    test = test.sort_values(["symbol", "date"]).copy()
    test["pos"] = test["p_up"] > cfg.threshold
    up = test["next_ret"] > 0
    accuracy = float(((test["p_up"] > 0.5) == up).mean())

    trades = []
    for _, g in test.groupby("symbol"):
        trades.extend(_trades_for(g, cfg.cost_per_side))
    stan_daily, hold_daily, exposure = daily_portfolio(test, cfg.cost_per_side)
    spy = test[test["symbol"] == "SPY"].set_index("date")["next_ret"]

    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    wins, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    profit_factor = None if not len(rets) else (99.0 if losses == 0 else round(float(wins / losses), 3))
    win_rate = float((rets > 0).mean()) if len(rets) else None
    stan_return = float((1 + stan_daily).prod() - 1)
    hold_return = float((1 + hold_daily).prod() - 1)
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
        "spy_return": round(float((1 + spy).prod() - 1), 4) if len(spy) else None,
        "exposure": round(exposure, 4),
        "max_dd": round(max_drawdown(stan_daily), 4),
        "hold_max_dd": round(max_drawdown(hold_daily), 4),
        # The bar: win at least half the trades AND make more than he loses, after costs.
        "passed": bool(win_rate is not None and win_rate >= 0.5 and (profit_factor or 0) > 1),
        "beat_hold": bool(stan_return > hold_return),
    }
    return exam, trades, {"stan": stan_daily, "hold": hold_daily, "spy": spy}


def walk_forward(
    panel: pd.DataFrame,
    cfg: StudyConfig = StudyConfig(),
    on_year: Callable[[dict, list[dict], int, int], None] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Train on the past, sit an exam on each following year. Returns (exams, trades, daily series)."""
    years = sorted(panel["date"].dt.year.unique())
    first = panel["date"].min()
    test_years = []
    for y in years:
        cutoff = pd.Timestamp(year=y, month=1, day=1) - pd.Timedelta(days=cfg.embargo_days)
        if (cutoff - first).days >= cfg.min_train_years * 365:
            test_years.append(y)

    exams, all_trades = [], []
    series = {"stan": [], "hold": [], "spy": []}
    for i, y in enumerate(test_years, start=1):
        cutoff = pd.Timestamp(year=y, month=1, day=1) - pd.Timedelta(days=cfg.embargo_days)
        train = panel[panel["date"] < cutoff]
        test = panel[panel["date"].dt.year == y].copy()
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=cfg.max_iter, learning_rate=0.05, random_state=7
        )
        model.fit(train[FEATURES], (train["next_ret"] > 0).astype(int))
        test["p_up"] = model.predict_proba(test[FEATURES])[:, 1]
        exam, trades, daily = grade_year(test, cfg)
        exams.append(exam)
        all_trades.extend(trades)
        for k in series:
            series[k].append(daily[k])
        if on_year:
            on_year(exam, trades, i, len(test_years))
    joined = {k: (pd.concat(v) if v else pd.Series(dtype=float)) for k, v in series.items()}
    return exams, all_trades, joined


def luck_test(
    trades: list[dict],
    closes: dict[str, pd.Series],
    cost: float,
    reps: int = 1000,
    opens: dict[str, pd.Series] | None = None,
    seed: int = 7,
) -> dict:
    """Would random entries have done as well? Each trade is replaced by a random
    entry in the same stock, same year, held the same number of days.

    Trades need: symbol, entry (signal date), days. With `opens`, entries are
    bought at the next day's open (matching rules that enter at the open).
    """
    rng = np.random.default_rng(seed)
    if not trades:
        return {"reps": 0, "actual_avg": None, "random_avg": None, "p_value": None,
                "actual_win": None, "random_win": None}
    sims, actual = [], []
    for t in trades:
        c = closes.get(t["symbol"])
        if c is None:
            continue
        idx = c.index
        year = pd.Timestamp(t["entry"]).year
        in_year = np.flatnonzero(idx.year == year)
        days = int(t["days"])
        hi = min(in_year.max() if len(in_year) else -1, len(c) - 1 - days)
        lo = in_year.min() if len(in_year) else 0
        if hi < lo:
            continue
        starts = rng.integers(lo, hi + 1, size=reps)
        cv = c.to_numpy()
        if opens is not None:
            ov = opens[t["symbol"]].to_numpy()
            sims.append(cv[starts + days] / ov[starts + 1] - 1 - 2 * cost)
        else:
            sims.append(cv[starts + days] / cv[starts] - 1 - 2 * cost)
        actual.append(t["ret"])
    if not sims:
        return {"reps": 0, "actual_avg": None, "random_avg": None, "p_value": None,
                "actual_win": None, "random_win": None}
    sims = np.vstack(sims)  # trades x reps
    actual = np.array(actual)
    rand_avg = sims.mean(axis=0)
    rand_win = (sims > 0).mean(axis=0)
    p = float((np.sum(rand_avg >= actual.mean()) + 1) / (reps + 1))
    return {
        "reps": reps,
        "actual_avg": round(float(actual.mean()), 5),
        "random_avg": round(float(rand_avg.mean()), 5),
        "random_avg_p95": round(float(np.quantile(rand_avg, 0.95)), 5),
        "p_value": round(p, 4),
        "actual_win": round(float((actual > 0).mean()), 4),
        "random_win": round(float(rand_win.mean()), 4),
    }


def summarize(exams: list[dict], trades: list[dict], daily: dict | None = None, luck: dict | None = None) -> dict:
    """Overall scorecard across every unseen year."""
    n = sum(e["n_calls"] for e in exams) or 1
    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    wins, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    stan_growth = float(np.prod([1 + e["stan_return"] for e in exams]))
    hold_growth = float(np.prod([1 + e["hold_return"] for e in exams]))
    spy_years = [e["spy_return"] for e in exams if e.get("spy_return") is not None]
    win_rate = float((rets > 0).mean()) if len(rets) else None
    pf = None if not len(rets) else (99.0 if losses == 0 else round(float(wins / losses), 3))
    exposure = (sum(e.get("exposure", 0) * e["n_calls"] for e in exams) / n) if exams else 0.0

    per_day_invested = per_day_hold = stan_dd = hold_dd = None
    if daily is not None and len(daily.get("stan", [])):
        stan_d, hold_d = daily["stan"], daily["hold"]
        stan_dd, hold_dd = round(max_drawdown(stan_d), 4), round(max_drawdown(hold_d), 4)
        if exposure > 0:
            per_day_invested = round(float(stan_d.mean()) / exposure, 6)
        per_day_hold = round(float(hold_d.mean()), 6)

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
        "spy_growth": round(float(np.prod([1 + r for r in spy_years])), 4) if spy_years else None,
        "exposure": round(exposure, 4),
        "per_day_invested": per_day_invested,
        "per_day_hold": per_day_hold,
        "max_dd": stan_dd,
        "hold_max_dd": hold_dd,
        "luck": luck,
        "ready_for_paper": ready,
    }


def find_lessons(trades: list[dict], cfg: StudyConfig = StudyConfig()) -> list[dict]:
    """Compare Stan's results across conditions. A condition only becomes a lesson
    when it clearly differs from his average:
    - mistake: it earns clearly less per trade than his average (by `lesson_margin`)
      AND either loses money or wins at least `lesson_gap` less often.
    - strength: it makes money, earns clearly more per trade than average, AND
      wins at least `lesson_gap` more often.
    """
    if not trades:
        return []
    t = pd.DataFrame(trades)
    overall_win = float((t["ret"] > 0).mean())
    overall_avg = float(t["ret"].mean())
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
    lessons = []
    for r in rows:
        evidence = {**r, "overall_win_rate": round(overall_win, 4), "overall_avg": round(overall_avg, 5)}
        earns_less = r["avg_return"] < overall_avg - cfg.lesson_margin
        earns_more = r["avg_return"] > overall_avg + cfg.lesson_margin
        losing = r["avg_return"] < 0
        wins_less = r["win_rate"] <= overall_win - cfg.lesson_gap
        wins_more = r["win_rate"] >= overall_win + cfg.lesson_gap
        if earns_less and (losing or wins_less):
            lessons.append({"kind": "mistake", "title": f"I do worse {r['when']}", "evidence": evidence})
        elif earns_more and r["avg_return"] > 0 and wins_more:
            lessons.append({"kind": "strength", "title": f"I do better {r['when']}", "evidence": evidence})
    lessons.sort(key=lambda l: l["evidence"]["avg_return"])
    return lessons
