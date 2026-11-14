"""Stan's first named strategy, and the stress tests it must survive.

Selloff bounce: when a stock's 14-day RSI closes under 30 (a sharp selloff),
buy, then sell when RSI recovers to 50 or after 10 trading days, whichever
comes first. By default Stan buys at the NEXT day's open, because he only knows
the RSI after the close.

The rule has no training step, so every year of history is a fair test year.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from agent.stan.study import luck_test, max_drawdown, rsi


@dataclass(frozen=True)
class SelloffRule:
    entry_rsi: float = 30.0
    exit_rsi: float = 50.0
    max_days: int = 10
    entry: str = "next_open"  # or "close"

    def describe(self) -> str:
        when = "next day's open" if self.entry == "next_open" else "same day's close"
        return (f"buy at the {when} after RSI closes under {self.entry_rsi:g}; sell when RSI "
                f"reaches {self.exit_rsi:g} or after {self.max_days} days")


BASE_RULE = SelloffRule()
BASE_COST = 0.001  # 0.10% per side: double the cost Stan's first term assumed
COST_LEVELS = (0.0005, 0.001, 0.0025)


def _clean(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.sort_index()
    return bars[~bars.index.duplicated(keep="last")].dropna(subset=["open", "close"])


def selloff_trades(bars: pd.DataFrame, symbol: str, rule: SelloffRule = BASE_RULE,
                   market_up: pd.Series | None = None) -> list[dict]:
    """Every trade the rule would have taken in one stock (gross of costs)."""
    bars = _clean(bars)
    c = bars["close"].astype(float).to_numpy()
    o = bars["open"].astype(float).to_numpy()
    dates = bars.index
    r = rsi(bars["close"].astype(float)).to_numpy()
    ma200 = (bars["close"] / bars["close"].rolling(200).mean() - 1).to_numpy()
    n = len(c)
    trades, i = [], 0
    while i < n - 1:
        if not np.isnan(r[i]) and r[i] < rule.entry_rsi:
            entry_price = o[i + 1] if rule.entry == "next_open" else c[i]
            if not entry_price or np.isnan(entry_price):
                i += 1
                continue
            j = i + 1
            while j < n - 1 and not (r[j] >= rule.exit_rsi) and (j - i) < rule.max_days:
                j += 1
            mkt = None
            if market_up is not None:
                v = market_up.get(dates[i])
                mkt = None if v is None or pd.isna(v) else bool(v)
            trades.append({
                "symbol": symbol, "entry": dates[i], "exit": dates[j], "days": int(j - i),
                "gross": float(c[j] / entry_price - 1),
                "above_ma200": bool(ma200[i] > 0) if not np.isnan(ma200[i]) else None,
                "market_up": mkt, "year": int(dates[i].year), "i": i, "j": j,
            })
            i = j + 1
        else:
            i += 1
    return trades


def _daily_curve(history: dict[str, pd.DataFrame], trades_by_symbol: dict[str, list[dict]],
                 rule: SelloffRule, cost: float) -> tuple[pd.Series, pd.Series, float]:
    """Equal money in every stock: (strategy daily returns, holding daily returns, time invested)."""
    strat_cols, hold_cols, held, total = [], [], 0, 0
    for symbol, bars in history.items():
        bars = _clean(bars)
        c = bars["close"].astype(float).to_numpy()
        o = bars["open"].astype(float).to_numpy()
        daily = np.zeros(len(c))
        in_pos = np.zeros(len(c), dtype=bool)
        for t in trades_by_symbol.get(symbol, []):
            i, j = t["i"], t["j"]
            if rule.entry == "next_open":
                daily[i + 1] += c[i + 1] / o[i + 1] - 1
                for k in range(i + 2, j + 1):
                    daily[k] += c[k] / c[k - 1] - 1
            else:
                for k in range(i + 1, j + 1):
                    daily[k] += c[k] / c[k - 1] - 1
            daily[i + 1] -= cost  # buy
            daily[j] -= cost      # sell
            in_pos[i + 1:j + 1] = True
        held += int(in_pos.sum())
        total += len(c)
        strat_cols.append(pd.Series(daily, index=bars.index, name=symbol))
        hold_cols.append(pd.Series(c, index=bars.index, name=symbol).pct_change())
    strat = pd.concat(strat_cols, axis=1)
    hold = pd.concat(hold_cols, axis=1)
    alive = hold.notna()
    strat_daily = strat.where(alive).mean(axis=1).fillna(0.0)
    hold_daily = hold.mean(axis=1).fillna(0.0)
    return strat_daily, hold_daily, (held / total if total else 0.0)


def score(trades: list[dict], cost: float) -> dict:
    net = np.array([t["gross"] - 2 * cost for t in trades])
    if not len(net):
        return {"trades": 0, "win_rate": None, "profit_factor": None, "avg": None, "t_stat": None}
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    sd = net.std(ddof=1) if len(net) > 1 else 0.0
    return {
        "trades": int(len(net)),
        "win_rate": round(float((net > 0).mean()), 4),
        "profit_factor": round(float(wins / losses), 3) if losses > 0 else 99.0,
        "avg": round(float(net.mean()), 5),
        "t_stat": round(float(net.mean() / (sd / np.sqrt(len(net)))), 2) if sd > 0 else None,
    }


def run_rule(history: dict[str, pd.DataFrame], rule: SelloffRule, cost: float,
             market_up: pd.Series | None = None, symbols: list[str] | None = None) -> dict:
    symbols = symbols or list(history)
    by_symbol = {s: selloff_trades(history[s], s, rule, market_up) for s in symbols if s in history}
    trades = [t for ts in by_symbol.values() for t in ts]
    for t in trades:
        t["ret"] = t["gross"] - 2 * cost
    return {"trades": trades, "by_symbol": by_symbol, "score": score(trades, cost)}


def stress_test(
    history: dict[str, pd.DataFrame],
    original_symbols: list[str],
    on_check=None,
    luck_reps: int = 1000,
) -> dict:
    """Run every stress test on the base rule. `on_check(name, passed, detail)` is called as each finishes."""
    spy = history.get("SPY")
    market_up = None
    if spy is not None:
        sc = _clean(spy)["close"].astype(float)
        ma = sc.rolling(200).mean()
        market_up = (sc > ma).astype(object).where(ma.notna(), None)

    base = run_rule(history, BASE_RULE, BASE_COST, market_up)
    trades, s = base["trades"], base["score"]
    checks: list[dict] = []

    def check(key: str, name: str, passed: bool, detail: str, data: dict | None = None) -> None:
        item = {"key": key, "name": name, "passed": bool(passed), "detail": detail, "data": data or {}}
        checks.append(item)
        if on_check:
            on_check(item)

    # 1. The basic bar.
    check("basic", "Wins at least half its trades and makes more than it loses (0.10% costs)",
          (s["win_rate"] or 0) >= 0.5 and (s["profit_factor"] or 0) > 1,
          f"{s['trades']} trades, won {_p(s['win_rate'])}, profit factor {s['profit_factor']}, "
          f"average {_sp(s['avg'], 2)} per trade", s)

    # 2. Year by year.
    years = {}
    for t in trades:
        years.setdefault(t["year"], []).append(t)
    by_year = [{"year": y, **score(ts, BASE_COST)} for y, ts in sorted(years.items())]
    good = [y for y in by_year if (y["avg"] or 0) > 0]
    share = len(good) / len(by_year) if by_year else 0
    worst = min(by_year, key=lambda y: y["avg"] or 0) if by_year else None
    check("years", "Makes money in at least 70% of years",
          share >= 0.7,
          f"made money in {len(good)} of {len(by_year)} years"
          + (f"; worst was {worst['year']} at {_sp(worst['avg'], 2)} per trade" if worst else ""),
          {"by_year": by_year})

    # 3. Luck test.
    closes = {k: _clean(v)["close"].astype(float) for k, v in history.items()}
    opens = {k: _clean(v)["open"].astype(float) for k, v in history.items()}
    luck = luck_test(trades, closes, BASE_COST, reps=luck_reps, opens=opens)
    check("luck", "Beats random entries in the same stocks and years (p under 0.05)",
          luck["p_value"] is not None and luck["p_value"] < 0.05,
          f"real trades averaged {_sp(luck['actual_avg'], 2)} vs {_sp(luck['random_avg'], 2)} for random "
          f"entries (p = {luck['p_value']}); random entries won {_p(luck['random_win'])}", luck)

    # 4. Costs.
    by_cost = []
    for cost in COST_LEVELS:
        sc_ = score(trades, cost)
        by_cost.append({"cost": cost, **sc_})
    high = by_cost[-1]
    check("costs", "Still profitable at 0.25% costs per side (wide spreads, bad fills)",
          (high["profit_factor"] or 0) > 1,
          "; ".join(f"{c['cost'] * 100:.2f}%: profit factor {c['profit_factor']}" for c in by_cost),
          {"by_cost": by_cost})

    # 5. Entry timing.
    at_close = run_rule(history, replace(BASE_RULE, entry="close"), BASE_COST)["score"]
    check("timing", "Works when buying at the next open, not only at the signal close",
          (s["profit_factor"] or 0) > 1,
          f"next open: profit factor {s['profit_factor']}; same close: {at_close['profit_factor']}",
          {"next_open": s, "close": at_close})

    # 6. Parameter grid: report every setting, choose none.
    grid = []
    for e in (25.0, 30.0, 35.0):
        for x in (50.0, 60.0):
            for d in (5, 10):
                rule = SelloffRule(e, x, d)
                grid.append({**asdict(rule), **run_rule(history, rule, BASE_COST)["score"]})
    ok = [g for g in grid if (g["profit_factor"] or 0) > 1]
    check("grid", "Profitable in at least 10 of 12 nearby settings (not a lucky setting)",
          len(ok) >= 10, f"{len(ok)} of {len(grid)} settings were profitable", {"grid": grid})

    # 7. New stocks only.
    new_syms = [k for k in history if k not in set(original_symbols)]
    new = run_rule(history, BASE_RULE, BASE_COST, symbols=new_syms)["score"] if new_syms else score([], BASE_COST)
    orig = run_rule(history, BASE_RULE, BASE_COST,
                    symbols=[k for k in original_symbols if k in history])["score"]
    check("new_stocks", "Works on stocks Stan never studied before",
          (new["profit_factor"] or 0) > 1 and (new["win_rate"] or 0) >= 0.5,
          f"{len(new_syms)} new stocks: won {_p(new['win_rate'])}, profit factor {new['profit_factor']}; "
          f"original {len(original_symbols)}: profit factor {orig['profit_factor']}",
          {"new": new, "original": orig})

    # 8. Market regime.
    up = score([t for t in trades if t["market_up"] is True], BASE_COST)
    down = score([t for t in trades if t["market_up"] is False], BASE_COST)
    check("regime", "Profitable both when the market is above and below its 200-day average",
          (up["profit_factor"] or 0) > 1 and (down["profit_factor"] or 0) > 1,
          f"market up: {up['trades']} trades, profit factor {up['profit_factor']}; "
          f"market down: {down['trades']} trades, profit factor {down['profit_factor']}",
          {"market_up": up, "market_down": down})

    # 9 and 10. Versus holding, fairly.
    strat_d, hold_d, exposure = _daily_curve(history, base["by_symbol"], BASE_RULE, BASE_COST)
    spy_d = closes["SPY"].pct_change().fillna(0.0) if "SPY" in closes else None
    per_day = float(strat_d.mean()) / exposure if exposure else 0.0
    hold_per_day = float(hold_d.mean())
    s_dd, h_dd = max_drawdown(strat_d), max_drawdown(hold_d)
    vs = {
        "growth": round(float((1 + strat_d).prod()), 4),
        "hold_growth": round(float((1 + hold_d).prod()), 4),
        "spy_growth": round(float((1 + spy_d).prod()), 4) if spy_d is not None else None,
        "exposure": round(exposure, 4),
        "per_day_invested": round(per_day, 6),
        "per_day_hold": round(hold_per_day, 6),
        "max_dd": round(s_dd, 4),
        "hold_max_dd": round(h_dd, 4),
    }
    check("per_day", "Earns more per day invested than holding does per day",
          per_day > hold_per_day,
          f"{_sp(per_day, 3)} per day invested vs {_sp(hold_per_day, 3)} per day holding "
          f"(invested {_p(exposure)} of the time)", vs)
    check("drawdown", "Smaller worst drop than holding",
          s_dd > h_dd, f"worst drop {_p(s_dd)} vs {_p(h_dd)} holding", vs)

    passed = sum(c["passed"] for c in checks)
    return {
        "name": "Selloff bounce",
        "rule": BASE_RULE.describe(),
        "cost": BASE_COST,
        "stocks": len(history),
        "score": s,
        "by_year": by_year,
        "versus": vs,
        "luck": luck,
        "checks": [{k: v for k, v in c.items() if k != "data"} for c in checks],
        "checks_passed": passed,
        "checks_total": len(checks),
        "verdict": "real_edge" if passed == len(checks) else ("promising" if passed >= len(checks) - 2 else "not_proven"),
    }


def _p(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _sp(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{digits}f}%"
