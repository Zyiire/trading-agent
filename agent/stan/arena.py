"""Stan's strategy arena: every strategy on the same daily history, ranked by risk tier.

Each strategy turns daily data into a daily return series after costs. Then every one is scored
the same way and placed in a tier by its MEASURED worst drop (not by its reputation):
    low     worst drop no deeper than 20%
    medium  20% to 40%
    high    deeper than 40%
Tier winners: the best Sharpe ratio in low and medium; the best yearly return in high (with a
Sharpe of 0.4 or better). Only "eligible" strategies can win: profitable over cash (positive
Sharpe) in both halves of the history and still positive at double costs, and not flagged for
survivorship bias.

Options are priced with Black-Scholes from the market's own volatility index (VIX for SPY, VXN for
QQQ), times 0.92 because at-the-money options trade a little below the index. There is no free
history of real option quotes, so these are estimates; costs of 2% of the premium are charged.

No lookahead: signals use closes up to the day before a trade. Cash earns the 13-week T-bill rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from agent.stan.portfolio import (ETFS, PortfolioConfig, momentum_decisions, month_ends,
                                  selloff_sleeve)
from agent.stan.strategies import _clean

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
NON_STOCKS = set(ETFS) | set(SECTORS) | {"EFA", "AGG", "IEF", "TLT", "GLD"}
EXTRAS = ["QQQ", "SPY", "EFA", "AGG", "IEF", "TLT", "GLD", *SECTORS, "^VIX", "^VXN", "^IRX"]


@dataclass(frozen=True)
class ArenaConfig:
    cost: float = 0.001          # 0.10% per side for stocks and ETFs
    opt_cost: float = 0.02       # 2% of an option's premium per trade
    iv_scale: float = 0.92       # at-the-money implied volatility vs the VIX/VXN level
    opt_days: int = 21           # one-month options, rolled every 21 trading days
    opt_budget: float = 0.05     # option buyers spend 5% of the account each month
    lev: float = 3.0             # leveraged ETF multiple
    lev_fee: float = 0.0095      # yearly fund fee for the leveraged ETF
    sma_days: int = 200
    trend_days: int = 210
    lag: int = 2                 # decide at close t, fill at close t+1, earn from t+2


# ---------------------------------------------------------------- data
@dataclass
class Data:
    close: pd.DataFrame          # adjusted closes, on SPY's calendar
    rf: pd.Series                # daily T-bill return
    vix: pd.Series | None
    vxn: pd.Series | None
    history: dict[str, pd.DataFrame]

    def has(self, *syms: str) -> bool:
        return all(s in self.close.columns and self.close[s].notna().sum() > 300 for s in syms)


def make_data(history: dict[str, pd.DataFrame]) -> Data:
    cal = _clean(history["SPY"]).index
    cols = {s: _clean(df)["close"].astype(float).reindex(cal) for s, df in history.items() if not s.startswith("^")}
    close = pd.DataFrame(cols, index=cal)
    irx = history.get("^IRX")
    rf = (_clean(irx)["close"].reindex(cal).ffill() / 100 / 252).fillna(0.0) if irx is not None else pd.Series(0.0, index=cal)
    vix = _clean(history["^VIX"])["close"].reindex(cal).ffill() / 100 if "^VIX" in history else None
    vxn = _clean(history["^VXN"])["close"].reindex(cal).ffill() / 100 if "^VXN" in history else None
    return Data(close, rf, vix, vxn, history)


def demo_extras(history: dict[str, pd.DataFrame], seed: int = 5) -> dict[str, pd.DataFrame]:
    """Made-up ETFs and indexes built from the demo market, so the arena runs offline."""
    rng = np.random.default_rng(seed)
    spy = _clean(history["SPY"])
    r = spy["close"].pct_change().fillna(0.0)
    out = {}

    def frame(close: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6})

    for sym, beta, noise in [("EFA", 0.9, 0.006), ("AGG", -0.05, 0.002), ("IEF", -0.1, 0.003), ("TLT", -0.25, 0.008),
                             ("GLD", 0.05, 0.009)] + [(s, rng.uniform(0.6, 1.3), 0.008) for s in SECTORS]:
        rr = beta * r + rng.normal(0.0001, noise, len(r))
        out[sym] = frame(100 * (1 + rr).cumprod())
    rv = r.rolling(21).std().bfill() * np.sqrt(252)
    out["^VIX"] = frame(100 * (rv * 1.2 + 0.02))
    out["^VXN"] = frame(100 * (rv * 1.45 + 0.03))
    out["^IRX"] = frame(pd.Series(2.0, index=spy.index))
    return out


def load_extras(years: int, demo: bool, history: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if demo:
        return demo_extras(history)
    from agent.stan.train import _from_yfinance  # noqa: PLC0415 (avoid a circular import)

    try:
        return _from_yfinance([s for s in EXTRAS if s not in history], years)
    except Exception:
        return {}


# ---------------------------------------------------------------- building blocks
def from_weights(w: pd.DataFrame, d: Data, cfg: ArenaConfig, cost: float | None = None) -> pd.Series:
    """Hold the decided weights with the lag; cash (1 - total weight) earns the T-bill rate."""
    cost = cfg.cost if cost is None else cost
    held = w.shift(cfg.lag).fillna(0.0)
    rets = d.close[held.columns].pct_change().fillna(0.0)
    r = (held * rets).sum(axis=1) + (1 - held.sum(axis=1)).clip(lower=0) * d.rf
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    return r - cost * turnover


def monthly(signal: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Keep a daily signal only at month ends, carried forward until the next one."""
    me = month_ends(signal.index)
    return signal.where(pd.Series(me, index=signal.index), np.nan).ffill() if isinstance(signal, pd.Series) \
        else signal[me].reindex(signal.index).ffill()


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(kind: str, s: float, k: float, t: float, vol: float, r: float = 0.0) -> float:
    """Black-Scholes price of a European call or put."""
    if t <= 0 or vol <= 0:
        return max(s - k, 0.0) if kind == "call" else max(k - s, 0.0)
    d1 = (math.log(s / k) + (r + vol * vol / 2) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if kind == "call":
        return s * _ncdf(d1) - k * math.exp(-r * t) * _ncdf(d2)
    return k * math.exp(-r * t) * _ncdf(-d2) - s * _ncdf(-d1)


def option_book(d: Data, cfg: ArenaConfig, legs_at: Callable[[int], list[tuple[str, str, float]]],
                mode: str = "buy", cost_mult: float = 1.0) -> pd.Series:
    """Roll one-month at-the-money options every `opt_days`, marked to market daily.

    legs_at(t) gives the legs to open at close t (using only data before t): (underlying, kind, share).
      mode "buy":      spend `share` of the account on premium for each leg; the rest sits in cash.
      mode "putwrite": sell puts on `share` of the account as notional, fully backed by cash.
      mode "covered":  hold the underlying with `share` of the account and sell that many calls.
    """
    close, idx = d.close, d.close.index
    equity, eq = 1.0, np.ones(len(idx))
    t = 0
    while t < len(idx) - 1:
        legs = legs_at(t)
        end = min(t + cfg.opt_days, len(idx) - 1)
        if not legs:
            for k in range(t + 1, end + 1):
                equity *= 1 + d.rf.iloc[k]
                eq[k] = equity
            t = end
            continue
        cash, pos = equity, []
        for sym, kind, share in legs:
            s0 = float(close[sym].iloc[t])
            vol_ix = d.vxn if sym == "QQQ" and d.vxn is not None else d.vix
            vol = float(vol_ix.iloc[t]) * cfg.iv_scale if vol_ix is not None else 0.2
            prem = bs(kind, s0, s0, cfg.opt_days / 252, vol, float(d.rf.iloc[t]) * 252)
            fee = cfg.opt_cost * cost_mult * prem
            if mode == "buy":
                n = share * equity / (prem + fee)
                cash -= n * (prem + fee)
                pos.append((sym, kind, s0, n, -1, 0.0))
            elif mode == "putwrite":
                n = share * equity / s0
                cash += n * (prem - fee)
                pos.append((sym, kind, s0, n, 1, 0.0))
            else:  # covered call: buy n shares, sell n calls
                n = share * equity / s0
                cash += n * (prem - fee) - n * s0 * (1 + cfg.cost * cost_mult)
                pos.append((sym, kind, s0, n, 1, n))
        for k in range(t + 1, end + 1):
            cash *= 1 + d.rf.iloc[k]
            tau = (end - k) / 252
            value = cash
            for sym, kind, strike, n, short, shares in pos:
                s = float(close[sym].iloc[k])
                vol_ix = d.vxn if sym == "QQQ" and d.vxn is not None else d.vix
                vol = float(vol_ix.iloc[k]) * cfg.iv_scale if vol_ix is not None else 0.2
                opt = bs(kind, s, strike, tau, vol, float(d.rf.iloc[k]) * 252)
                value += shares * s - short * n * opt if short == 1 else n * opt
            eq[k] = value
        for sym, kind, strike, n, short, shares in pos:  # covered calls sell the shares at the roll
            if shares:
                eq[end] -= shares * float(close[sym].iloc[end]) * cfg.cost * cost_mult
        equity = eq[end]
        t = end
    s = pd.Series(eq, index=idx)
    return s.pct_change().fillna(0.0)


def above_avg(px: pd.Series, days: int) -> pd.Series:
    """True when yesterday's close was above its `days`-day average (known before today's trade)."""
    return (px > px.rolling(days).mean()).shift(1).fillna(False).astype(bool)


# ---------------------------------------------------------------- the strategies
@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    how: str
    evidence: str
    needs: tuple[str, ...]
    run: Callable[[Data, ArenaConfig, float], pd.Series]   # (data, config, cost multiplier) -> daily returns
    survivorship: bool = False


def _w(d: Data, cols: dict[str, pd.Series | float]) -> pd.DataFrame:
    return pd.DataFrame({k: (v if isinstance(v, pd.Series) else pd.Series(v, index=d.close.index))
                         for k, v in cols.items()}, index=d.close.index).astype(float)


def s_hold(d, cfg, m):
    return from_weights(_w(d, {"SPY": 1.0}), d, cfg, cfg.cost * m)


def s_6040(d, cfg, m):
    return from_weights(_w(d, {"SPY": 0.6, "AGG": 0.4}), d, cfg, cfg.cost * m)


def s_trend(d, cfg, m):
    px = d.close["SPY"]
    on = monthly((px > px.rolling(cfg.trend_days).mean()).astype(float)).fillna(0.0)
    return from_weights(_w(d, {"SPY": on}), d, cfg, cfg.cost * m)


def s_voltarget(d, cfg, m):
    px = d.close["SPY"]
    rv = px.pct_change().rolling(21).std() * np.sqrt(252)
    w = monthly((0.15 / rv).clip(upper=1.0)).fillna(1.0)
    return from_weights(_w(d, {"SPY": w}), d, cfg, cfg.cost * m)


def s_gem(d, cfg, m):
    """Antonacci's Global Equities Momentum: SPY or EFA by 12-month return if SPY beat T-bills, else AGG."""
    c = d.close
    r12 = c[["SPY", "EFA"]] / c[["SPY", "EFA"]].shift(252) - 1
    bill12 = (1 + d.rf).rolling(252).apply(np.prod, raw=True) - 1
    pick = pd.DataFrame(0.0, index=c.index, columns=["SPY", "EFA", "AGG"])
    risk_on = r12["SPY"] > bill12
    pick.loc[risk_on & (r12["SPY"] >= r12["EFA"]), "SPY"] = 1.0
    pick.loc[risk_on & (r12["EFA"] > r12["SPY"]), "EFA"] = 1.0
    pick.loc[~risk_on & r12["SPY"].notna(), "AGG"] = 1.0
    return from_weights(monthly(pick).fillna(0.0), d, cfg, cfg.cost * m)


def s_sectors(d, cfg, m):
    """Top 3 of the 9 sector ETFs by 12-1 month return, only if positive; the rest in cash."""
    c = d.close[SECTORS]
    mom = c.shift(21) / c.shift(252) - 1
    w = pd.DataFrame(0.0, index=c.index, columns=SECTORS)
    for t in range(len(c)):
        row = mom.iloc[t].dropna()
        row = row[row > 0].nlargest(3)
        if len(row):
            w.iloc[t, [SECTORS.index(x) for x in row.index]] = 1 / 3
    return from_weights(monthly(w).fillna(0.0), d, cfg, cfg.cost * m)


def s_putwrite(d, cfg, m):
    return option_book(d, cfg, lambda t: [("SPY", "put", 1.0)], "putwrite", m)


def s_covered(d, cfg, m):
    return option_book(d, cfg, lambda t: [("SPY", "call", 1.0)], "covered", m)


def s_selloff(d, cfg, m):
    pc = PortfolioConfig(cost=cfg.cost * m)
    syms = [s for s in d.history if s not in NON_STOCKS and not s.startswith("^")]
    r = selloff_sleeve(d.history, d.close.index, pc, symbols=syms)
    return r  # cash in idle slots earns nothing here (matches Stan's paper account)


def s_momentum(d, cfg, m):
    stocks = d.close[[s for s in d.close.columns if s not in NON_STOCKS]]
    w = momentum_decisions(stocks, PortfolioConfig())
    return from_weights(w, d, cfg, cfg.cost * m)


def _lev(d: Data, cfg: ArenaConfig, on: pd.Series, m: float) -> pd.Series:
    """A daily-reset 3x QQQ fund (fee and borrowing included), held when `on`, else T-bills."""
    r = d.close["QQQ"].pct_change().fillna(0.0)
    lev = cfg.lev * r - (cfg.lev - 1) * d.rf - cfg.lev_fee / 252
    held = on.astype(float)
    switch = held.diff().abs().fillna(held.abs())
    return held * lev + (1 - held) * d.rf - cfg.cost * m * switch


def s_lev_hold(d, cfg, m):
    return _lev(d, cfg, pd.Series(True, index=d.close.index), m)


def s_lev_trend(d, cfg, m):
    """Gayed & Bilello's leverage rotation: 3x QQQ while QQQ is above its 200-day average, else T-bills."""
    return _lev(d, cfg, above_avg(d.close["QQQ"], cfg.sma_days).shift(1).fillna(False), m)


def s_calls(d, cfg, m):
    return option_book(d, cfg, lambda t: [("SPY", "call", cfg.opt_budget)], "buy", m)


def s_puts(d, cfg, m):
    return option_book(d, cfg, lambda t: [("QQQ", "put", cfg.opt_budget)], "buy", m)


def s_pair(d, cfg, m):
    return option_book(d, cfg, lambda t: [("SPY", "call", cfg.opt_budget / 2), ("QQQ", "put", cfg.opt_budget / 2)],
                       "buy", m)


def s_trend_options(d, cfg, m):
    spy_up = above_avg(d.close["SPY"], cfg.trend_days)
    qqq_up = above_avg(d.close["QQQ"], cfg.trend_days)

    def legs(t):
        out = []
        if spy_up.iloc[t]:
            out.append(("SPY", "call", cfg.opt_budget))
        elif not qqq_up.iloc[t]:
            out.append(("QQQ", "put", cfg.opt_budget))
        return out
    return option_book(d, cfg, legs, "buy", m)


STRATEGIES = [
    Strategy("hold_spy", "Hold SPY", "buy the S&P 500 and hold it", "the benchmark every strategy has to beat",
             ("SPY",), s_hold),
    Strategy("sixty_forty", "60/40 stocks and bonds", "60% SPY, 40% AGG bonds, rebalanced daily",
             "the classic balanced portfolio", ("SPY", "AGG"), s_6040),
    Strategy("trend_core", "Trend core", "SPY while above its 10-month average at a month end, else T-bills",
             "a century of trend-following evidence (AQR)", ("SPY",), s_trend),
    Strategy("vol_target", "Volatility target", "SPY sized to 15% volatility, monthly",
             "volatility-managed portfolios (Moreira & Muir)", ("SPY",), s_voltarget),
    Strategy("gem", "Dual momentum (GEM)", "SPY or EFA, whichever rose more in 12 months, if SPY beat T-bills; "
             "else AGG bonds; monthly", "Antonacci's Global Equities Momentum", ("SPY", "EFA", "AGG"), s_gem),
    Strategy("sectors", "Sector momentum", "top 3 of 9 sector ETFs by 12-1 month return, if positive; monthly",
             "momentum across industries", tuple(SECTORS), s_sectors),
    Strategy("putwrite", "Sell SPY puts (cash-secured)", "sell a 1-month at-the-money SPY put every month, "
             "backed fully by cash", "the Cboe PutWrite index and the volatility risk premium", ("SPY",), s_putwrite),
    Strategy("covered_call", "Covered calls on SPY", "hold SPY and sell a 1-month at-the-money call every month",
             "the Cboe BuyWrite index", ("SPY",), s_covered),
    Strategy("selloff", "Selloff bounce", "Stan's rule: buy after RSI under 30, sell at 50 or 10 days; 10% a position",
             "short-term reversal; passed 9 of 10 of Stan's stress tests", ("SPY",), s_selloff, survivorship=True),
    Strategy("stock_momentum", "Stock momentum", "top 5 of Stan's stocks by 12-1 month return, if positive; monthly",
             "Jegadeesh & Titman momentum", ("SPY",), s_momentum, survivorship=True),
    Strategy("lev_trend", "3x QQQ with a 200-day trend switch", "a 3x Nasdaq fund while QQQ is above its 200-day "
             "average, else T-bills", "Gayed & Bilello, Leverage for the Long Run", ("QQQ",), s_lev_trend),
    Strategy("lev_hold", "3x QQQ, always", "hold a 3x Nasdaq fund (like TQQQ) no matter what",
             "no edge: shows what raw leverage does", ("QQQ",), s_lev_hold),
    Strategy("buy_calls", "Buy SPY calls", "spend 5% a month on 1-month at-the-money SPY calls, rest in T-bills",
             "option buyers pay the volatility risk premium", ("SPY",), s_calls),
    Strategy("buy_puts", "Buy QQQ puts", "spend 5% a month on 1-month at-the-money QQQ puts, rest in T-bills",
             "option buyers pay the volatility risk premium", ("QQQ",), s_puts),
    Strategy("calls_and_puts", "SPY calls + QQQ puts together", "2.5% a month on SPY calls and 2.5% on QQQ puts",
             "Zy's idea: profits when SPY rises or QQQ falls", ("SPY", "QQQ"), s_pair),
    Strategy("trend_options", "Calls in uptrends, puts in downtrends", "5% a month: SPY calls when SPY is above "
             "its 10-month average, QQQ puts when QQQ is below it, else nothing", "Zy's idea with a trend filter",
             ("SPY", "QQQ"), s_trend_options),
]


# ---------------------------------------------------------------- scoring and ranking
def stats(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 60:
        return {"cagr": None, "vol": None, "sharpe": None, "max_dd": None, "growth": None, "worst_year": None}
    ex = r - rf.reindex(r.index).fillna(0.0)
    eq = (1 + r).cumprod()
    vol = float(r.std() * np.sqrt(252))
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    return {"cagr": round(float(eq.iloc[-1] ** (252 / len(r)) - 1), 4), "vol": round(vol, 4),
            "sharpe": round(float(ex.mean() * 252 / vol), 3) if vol > 0 else None,
            "max_dd": round(float((eq / eq.cummax() - 1).min()), 4), "growth": round(float(eq.iloc[-1]), 3),
            "worst_year": round(float(by_year.min()), 4)}


def tier_of(max_dd: float | None) -> str:
    if max_dd is None:
        return "n/a"
    return "low" if max_dd >= -0.20 else ("medium" if max_dd >= -0.40 else "high")


def run_arena(history: dict[str, pd.DataFrame], cfg: ArenaConfig = ArenaConfig(),
              on_result: Callable[[dict], None] | None = None) -> dict:
    d = make_data(history)
    start = 260 + cfg.lag   # every strategy has a full year of lookback from here
    results = []
    for st in STRATEGIES:
        if not d.has(*st.needs) or (st.key in ("putwrite", "covered_call", "buy_calls", "calls_and_puts",
                                               "trend_options") and d.vix is None):
            results.append({"key": st.key, "name": st.name, "how": st.how, "evidence": st.evidence,
                            "skipped": "missing data"})
            continue
        r = st.run(d, cfg, 1.0).iloc[start:]
        r2 = st.run(d, cfg, 2.0).iloc[start:]
        s = stats(r, d.rf)
        mid = len(r) // 2
        h1, h2 = stats(r.iloc[:mid], d.rf), stats(r.iloc[mid:], d.rf)
        s2 = stats(r2, d.rf)
        eligible = all(x["sharpe"] is not None and x["sharpe"] > 0 for x in (h1, h2, s2)) and not st.survivorship
        res = {"key": st.key, "name": st.name, "how": st.how, "evidence": st.evidence, **s,
               "sharpe_halves": [h1["sharpe"], h2["sharpe"]], "sharpe_2x_costs": s2["sharpe"],
               "tier": tier_of(s["max_dd"]), "eligible": eligible, "survivorship": st.survivorship}
        results.append(res)
        if on_result:
            on_result(res)
    ran = [x for x in results if "skipped" not in x]
    spy = next((x for x in ran if x["key"] == "hold_spy"), None)
    winners = {}
    for tier in ("low", "medium", "high"):
        pool = [x for x in ran if x["tier"] == tier and x["eligible"] and x["key"] != "hold_spy"]
        if tier == "high":
            pool = [x for x in pool if (x["sharpe"] or 0) >= 0.4]
            pool.sort(key=lambda x: -(x["cagr"] or -9))
        else:
            pool.sort(key=lambda x: -(x["sharpe"] or -9))
        winners[tier] = pool[0]["key"] if pool else None
    sized = None
    if winners["low"] is None:
        # risk is mostly position size: run the best eligible strategy smaller, rest in T-bills
        best = max((x for x in ran if x["eligible"] and x["key"] != "hold_spy" and x["max_dd"]),
                   key=lambda x: x["sharpe"] or -9, default=None)
        if best:
            st = next(z for z in STRATEGIES if z.key == best["key"])
            r = st.run(d, cfg, 1.0).iloc[start:]
            rf = d.rf.iloc[start:]
            frac = min(1.0, 0.18 / abs(best["max_dd"]))
            for _ in range(10):
                mix = frac * r + (1 - frac) * rf
                s_mix = stats(mix, d.rf)
                if s_mix["max_dd"] >= -0.20:
                    break
                frac *= 0.9
            sized = {"key": f"{best['key']}_sized", "name": f"{best['name']} at {frac:.0%} size",
                     "how": f"{best['how']}; only {frac:.0%} of the account, the rest in T-bills",
                     "evidence": best["evidence"], **s_mix, "base": best["key"], "fraction": round(frac, 3),
                     "tier": "low", "eligible": True, "survivorship": False,
                     "sharpe_halves": best["sharpe_halves"], "sharpe_2x_costs": best["sharpe_2x_costs"]}
            ran.append(sized)
            winners["low"] = sized["key"]
            if on_result:
                on_result(sized)
    ranked = sorted(ran, key=lambda x: -(x["sharpe"] or -9))
    for i, x in enumerate(ranked, 1):
        x["rank"] = i
        x["beats_spy"] = bool(spy and x["sharpe"] is not None and spy["sharpe"] is not None
                              and x["sharpe"] > spy["sharpe"])
    idx = d.close.index[start:]
    return {"name": "Strategy arena", "first_day": str(idx[0].date()), "last_day": str(idx[-1].date()),
            "strategies": ranked + [x for x in results if "skipped" in x], "winners": winners,
            "spy": spy, "cost": cfg.cost, "opt_cost": cfg.opt_cost,
            "notes": ["Options are priced from VIX/VXN with Black-Scholes, not real quotes, so treat their results "
                      "as estimates.",
                      "Stock momentum and the selloff bounce use today's well-known stocks, so they can't win a tier.",
                      "Tiers come from each strategy's measured worst drop: low up to 20%, medium 20-40%, "
                      "high deeper than 40%."]}
