"""Stan's portfolio: four documented edges, added one stage at a time, each kept only if it helps.

    Stage 1  Trend core          hold SPY while it closes above its 10-month (210-day) average at a
                                 month end; otherwise cash. (Time-series momentum / trend following.)
    Stage 2  Volatility sizing   at each month end, hold less SPY when the last month was wild:
                                 exposure = min(1, 15% / last 21 days' annualised volatility).
    Stage 3  Selloff bounce      Stan's own tested rule, 10% a position, at most 10 at once.
    Stage 4  Momentum            at each month end, the 5 stocks with the best return from 12 months
                                 to 1 month ago, if that return is positive; equal weights.

Allocations are fixed in advance, not tuned: the SPY part takes what's left after the sleeves
(30% for the selloff bounce, 30% for momentum, when kept).

A stage is kept only if, against the portfolio without it:
  * its Sharpe ratio is no worse (within 0.02) over the whole period AND in each half, and
  * it either raises the Sharpe ratio by 0.02+ or makes the worst drop 5+ points shallower.

No lookahead: a month-end decision uses closes up to that day, the trade fills at the next day's
close, and the first return it earns is the day after that. Cash earns nothing (conservative).
Costs: 0.10% per side on every change in weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from agent.stan.strategies import BASE_RULE, _clean, selloff_trades

COST = 0.001
ETFS = {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLK", "SMH"}


@dataclass(frozen=True)
class PortfolioConfig:
    market: str = "SPY"
    trend_days: int = 210          # about 10 months
    vol_days: int = 21
    target_vol: float = 0.15
    mom_long: int = 252            # 12 months back...
    mom_skip: int = 21             # ...to 1 month back
    mom_top: int = 5
    selloff_slot: float = 0.10     # 10% of the sleeve per selloff position
    selloff_max: int = 10
    sleeve_alloc: float = 0.30     # each kept sleeve's share of the portfolio
    lag: int = 2                   # decide at close t, fill at close t+1, earn from t+2
    cost: float = COST


# ---------------------------------------------------------------- data
def closes_frame(history: dict[str, pd.DataFrame], market: str) -> pd.DataFrame:
    """Closes on the market's trading calendar (no filling across missing days)."""
    cal = _clean(history[market]).index
    cols = {s: _clean(df)["close"].astype(float).reindex(cal) for s, df in history.items()}
    return pd.DataFrame(cols, index=cal)


def month_ends(index: pd.DatetimeIndex) -> np.ndarray:
    m = index.to_period("M")
    return np.append(m[1:] != m[:-1], True)


# ---------------------------------------------------------------- sleeves: decided weights
def trend_decisions(px: pd.Series, cfg: PortfolioConfig) -> pd.Series:
    """1 while above the long average at the last month end, else 0 (decided at month-end closes)."""
    above = (px > px.rolling(cfg.trend_days).mean()).astype(float)
    above[px.rolling(cfg.trend_days).mean().isna()] = np.nan
    me = pd.Series(month_ends(px.index), index=px.index)
    return above.where(me).ffill().fillna(0.0)


def vol_decisions(px: pd.Series, cfg: PortfolioConfig) -> pd.Series:
    rv = px.pct_change().rolling(cfg.vol_days).std() * np.sqrt(252)
    scale = (cfg.target_vol / rv).clip(upper=1.0)
    me = pd.Series(month_ends(px.index), index=px.index)
    return scale.where(me).ffill().fillna(1.0)


def momentum_decisions(closes: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    stocks = [c for c in closes.columns if c not in ETFS]
    px = closes[stocks]
    mom = px.shift(cfg.mom_skip) / px.shift(cfg.mom_long) - 1
    me = month_ends(closes.index)
    w = pd.DataFrame(np.nan, index=closes.index, columns=stocks)
    for t in np.flatnonzero(me):
        row = mom.iloc[t].dropna()
        row = row[row > 0].nlargest(cfg.mom_top)
        w.iloc[t] = 0.0
        if len(row):
            w.iloc[t, [stocks.index(s) for s in row.index]] = 1.0 / cfg.mom_top
    return w.ffill().fillna(0.0)


def returns_from_weights(decided: pd.DataFrame, rets: pd.DataFrame, cfg: PortfolioConfig) -> pd.Series:
    """Daily return of holding the decided weights with the lag, minus costs on weight changes."""
    held = decided.shift(cfg.lag).fillna(0.0)
    r = (held * rets[held.columns].fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    return r - cfg.cost * turnover


def selloff_sleeve(history: dict[str, pd.DataFrame], index: pd.DatetimeIndex, cfg: PortfolioConfig,
                   symbols: list[str] | None = None) -> pd.Series:
    """Stan's selloff bounce as a sleeve: 10% per position, at most 10 open, first come first served.
    Buys at the open after the signal (as the paper account does), sells at the exit day's close."""
    events = []
    for sym, bars in history.items():
        if symbols is not None and sym not in symbols:
            continue
        b = _clean(bars)
        for t in selloff_trades(b, sym, BASE_RULE):
            events.append((b.index[t["i"] + 1], b.index[t["j"]], sym))
    events.sort()
    out = pd.Series(0.0, index=index)
    open_until: list[pd.Timestamp] = []
    pos = {d: k for k, d in enumerate(index)}
    for start, end, sym in events:
        open_until = [e for e in open_until if e >= start]
        if len(open_until) >= cfg.selloff_max or start not in pos or end not in pos:
            continue
        open_until.append(end)
        b = _clean(history[sym])
        o, c = b["open"].astype(float), b["close"].astype(float)
        days = b.index[(b.index >= start) & (b.index <= end)]
        for k, d in enumerate(days):
            r = c[d] / o[d] - 1 if k == 0 else c[d] / c[days[k - 1]] - 1
            out[d] += cfg.selloff_slot * r
        out[start] -= cfg.selloff_slot * cfg.cost
        out[end] -= cfg.selloff_slot * cfg.cost
    return out


# ---------------------------------------------------------------- measuring
def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"cagr": None, "vol": None, "sharpe": None, "max_dd": None, "growth": None, "worst_year": None}
    eq = (1 + r).cumprod()
    years = len(r) / 252
    vol = float(r.std() * np.sqrt(252))
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    return {"cagr": round(float(eq.iloc[-1] ** (1 / years) - 1), 4), "vol": round(vol, 4),
            "sharpe": round(float(r.mean() * 252 / vol), 3) if vol > 0 else None,
            "max_dd": round(float((eq / eq.cummax() - 1).min()), 4), "growth": round(float(eq.iloc[-1]), 3),
            "worst_year": round(float(by_year.min()), 4)}


def halves(r: pd.Series) -> tuple[pd.Series, pd.Series]:
    mid = len(r) // 2
    return r.iloc[:mid], r.iloc[mid:]


def judge(new: pd.Series, base: pd.Series) -> tuple[bool, str]:
    """The keep rule from the module docstring."""
    n, b = stats(new), stats(base)
    (n1, n2), (b1, b2) = [stats(x) for x in halves(new)], [stats(x) for x in halves(base)]
    no_worse = all(x["sharpe"] is not None and y["sharpe"] is not None and x["sharpe"] >= y["sharpe"] - 0.02
                   for x, y in ((n, b), (n1, b1), (n2, b2)))
    helps = (n["sharpe"] or 0) >= (b["sharpe"] or 0) + 0.02 or (n["max_dd"] or -1) >= (b["max_dd"] or -1) + 0.05
    why = (f"Sharpe {b['sharpe']} -> {n['sharpe']} (first half {b1['sharpe']} -> {n1['sharpe']}, second half "
           f"{b2['sharpe']} -> {n2['sharpe']}); worst drop {_pct(b['max_dd'])} -> {_pct(n['max_dd'])}; "
           f"yearly return {_pct(b['cagr'])} -> {_pct(n['cagr'])}")
    return bool(no_worse and helps), why


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


# ---------------------------------------------------------------- the staged build
def build(history: dict[str, pd.DataFrame], cfg: PortfolioConfig = PortfolioConfig(),
          on_check: Callable[[dict], None] | None = None) -> dict:
    closes = closes_frame(history, cfg.market)
    rets = closes.pct_change()
    mkt = closes[cfg.market]
    idx = closes.index
    start = cfg.mom_long + cfg.lag  # every sleeve has its full lookback from here
    spy = rets[cfg.market].fillna(0.0)

    trend = trend_decisions(mkt, cfg)
    vol = vol_decisions(mkt, cfg)
    sleeves = {
        "selloff": selloff_sleeve(history, idx, cfg),
        "momentum": returns_from_weights(momentum_decisions(closes, cfg), rets, cfg),
    }

    def core(use_trend: bool, use_vol: bool) -> pd.Series:
        w = pd.Series(1.0, index=idx)
        if use_trend:
            w = w * trend
        if use_vol:
            w = w * vol
        return returns_from_weights(w.to_frame(cfg.market), rets, cfg)

    def mix(use_trend, use_vol, kept_sleeves) -> pd.Series:
        a = cfg.sleeve_alloc
        out = (1 - a * len(kept_sleeves)) * core(use_trend, use_vol)
        for s in kept_sleeves:
            out = out + a * sleeves[s]
        return out.iloc[start:]

    checks, stages = [], []
    state = {"trend": False, "vol": False, "sleeves": []}

    def record(key, name, cand_state, label):
        base = mix(state["trend"], state["vol"], state["sleeves"])
        cand = mix(cand_state["trend"], cand_state["vol"], cand_state["sleeves"])
        kept, why = judge(cand, base)
        item = {"key": key, "name": name, "passed": kept, "detail": ("kept: " if kept else "left out: ") + why}
        checks.append(item)
        stages.append({"key": key, "label": label, "kept": kept, **stats(cand)})
        if on_check:
            on_check(item)
        if kept:
            state.update(cand_state)

    hold = spy.iloc[start:]
    record("trend", "Stage 1, trend core: hold SPY only above its 10-month average",
           {**state, "trend": True}, "Trend core")
    record("vol", "Stage 2, volatility sizing: hold less when the last month was wild",
           {**state, "vol": True}, "Volatility sizing")
    record("selloff", "Stage 3, selloff bounce sleeve (30%)",
           {**state, "sleeves": state["sleeves"] + ["selloff"]}, "Selloff bounce")
    record("momentum", "Stage 4, momentum sleeve: top 5 stocks by 12-1 month return (30%)",
           {**state, "sleeves": state["sleeves"] + ["momentum"]}, "Momentum")

    final = mix(state["trend"], state["vol"], state["sleeves"])
    f, h = stats(final), stats(hold)
    (f1, f2), (h1, h2) = [stats(x) for x in halves(final)], [stats(x) for x in halves(hold)]
    beat_sharpe = all((x["sharpe"] or -9) > (y["sharpe"] or 9) for x, y in ((f, h), (f1, h1), (f2, h2)))
    item = {"key": "final_sharpe", "name": "Final portfolio: better return for its risk than holding SPY, in both halves",
            "passed": beat_sharpe,
            "detail": f"Sharpe {f['sharpe']} vs SPY {h['sharpe']} (halves {f1['sharpe']}/{f2['sharpe']} vs "
                      f"{h1['sharpe']}/{h2['sharpe']}); {_pct(f['cagr'])} a year vs {_pct(h['cagr'])}"}
    checks.append(item)
    if on_check:
        on_check(item)
    shallower = f["max_dd"] is not None and h["max_dd"] is not None and f["max_dd"] >= h["max_dd"] * 2 / 3
    item = {"key": "final_drawdown", "name": "Final portfolio: worst drop no more than two-thirds of SPY's",
            "passed": shallower, "detail": f"worst drop {_pct(f['max_dd'])} vs SPY {_pct(h['max_dd'])}"}
    checks.append(item)
    if on_check:
        on_check(item)

    curve = (1 + final).cumprod()
    spy_curve = (1 + hold).cumprod()
    step = max(1, len(curve) // 120)
    passed = sum(c["passed"] for c in checks[-2:])
    kept_labels = [s["label"] for s in stages if s["kept"]]
    return {
        "name": "Stan's portfolio",
        "rule": "the stages kept: " + (", ".join(kept_labels) if kept_labels else "none, so it's simply SPY held"),
        "unit": "portfolio",
        "cost": cfg.cost,
        "stocks": len(closes.columns),
        "first_day": str(final.index[0].date()),
        "last_day": str(final.index[-1].date()),
        "spy": h,
        "final": f,
        "stages": stages,
        "kept": kept_labels,
        "curve": [{"date": str(d.date()), "stan": round(float(curve[d]), 4), "spy": round(float(spy_curve[d]), 4)}
                  for d in curve.index[::step]],
        "checks": checks,
        "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks),
        "survivorship_note": ("the 43 stocks are today's well-known names, so momentum and the selloff bounce look "
                              "better on this history than they would have in real time"),
        "verdict": "real_edge" if passed == 2 else ("promising" if passed == 1 else "not_proven"),
    }
