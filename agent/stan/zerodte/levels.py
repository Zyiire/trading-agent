"""The day's key levels for the 0DTE bot, from everything the pipeline collects.

  round     $10 round numbers (TZ's 680), and $5 as minor ones
  prior     yesterday's high, low and close; today's open
  gex       call wall, put wall and the zero-gamma flip (gex.py)
  oi        the strikes with the most open interest, calls and puts

Levels closer together than 0.1% are one level that keeps every reason, so a $10 round number that
is also the call wall shows as one stronger level. Phase 3 tests which kinds actually matter.
"""
from __future__ import annotations

from agent.stan.zerodte import gex
from agent.stan.zerodte.config import ZeroDteConfig

WEIGHT = {"round10": 2, "round5": 1, "prior_high": 2, "prior_low": 2, "prior_close": 1, "open": 1,
          "call_wall": 3, "put_wall": 3, "zero_gamma": 2, "call_oi": 1, "put_oi": 1}


def round_numbers(spot: float, cfg: ZeroDteConfig) -> list[tuple[float, str]]:
    lo, hi = spot * (1 - cfg.level_window), spot * (1 + cfg.level_window)
    out = []
    k = int(lo // cfg.minor_round_step) * cfg.minor_round_step
    while k <= hi:
        if k >= lo:
            major = abs(k / cfg.round_step - round(k / cfg.round_step)) < 1e-9
            out.append((float(k), "round10" if major else "round5"))
        k += cfg.minor_round_step
    return out


def top_oi(rows, kind: str, n: int) -> list[float]:
    by: dict[float, float] = {}
    for r in rows:
        if r.type == kind and r.oi:
            by[r.strike] = by.get(r.strike, 0.0) + r.oi
    return sorted(by, key=by.get, reverse=True)[:n]


def key_levels(spot: float, prior_bars: list[dict] | None, today_bars: list[dict] | None, rows, now_ts: int,
               cfg: ZeroDteConfig | None = None, iv_fallback: float | None = None) -> dict:
    cfg = cfg or ZeroDteConfig()
    raw: list[tuple[float, str]] = round_numbers(spot, cfg)
    if prior_bars:
        raw += [(max(b["h"] for b in prior_bars), "prior_high"), (min(b["l"] for b in prior_bars), "prior_low"),
                (prior_bars[-1]["c"], "prior_close")]
    if today_bars:
        raw.append((today_bars[0]["o"], "open"))
    g = gex.summary(rows, spot, now_ts, iv_fallback) if rows else gex.summary([], spot, now_ts)
    for key in ("call_wall", "put_wall", "zero_gamma"):
        if g[key] is not None:
            raw.append((float(g[key]), key))
    for kind in ("call", "put"):
        raw += [(k, f"{kind}_oi") for k in top_oi(rows or [], kind, cfg.top_oi_strikes)]
    lo, hi = spot * (1 - cfg.level_window), spot * (1 + cfg.level_window)
    return {"spot": spot, "gex": g, "levels": merge([x for x in raw if lo <= x[0] <= hi], spot, cfg.level_merge)}


def merge(raw: list[tuple[float, str]], spot: float, tol: float) -> list[dict]:
    groups: list[list[tuple[float, str]]] = []
    for price, kind in sorted(raw):
        if groups and price - groups[-1][0][0] <= spot * tol:
            groups[-1].append((price, kind))
        else:
            groups.append([(price, kind)])
    out = []
    for grp in groups:
        anchors = [p for p, k in grp if k.startswith("round") or k.endswith("_oi") or k.endswith("_wall")]
        price = anchors[0] if anchors else sum(p for p, _ in grp) / len(grp)
        kinds = sorted({k for _, k in grp}, key=lambda k: -WEIGHT[k])
        out.append({"price": round(price, 2), "kinds": kinds, "strength": sum(WEIGHT[k] for k in set(kinds)),
                    "side": "above" if price > spot else "below" if price < spot else "at"})
    return out
