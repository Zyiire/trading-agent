"""4. Stops, size, targets, and walking a trade bar by bar."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agent.stan.price_action.config import PriceActionConfig


@dataclass(frozen=True)
class Plan:
    side: int
    entry: float
    stop: float
    target: float
    risk: float          # per share, entry to stop
    shares: int          # 0 when the account can't afford one share at this risk
    target_kind: str     # "rr" or "level"


def risk_budget(cfg: PriceActionConfig) -> float:
    return cfg.risk_dollars if cfg.risk_dollars is not None else cfg.account * cfg.risk_pct


def size(entry: float, risk: float, cfg: PriceActionConfig) -> int:
    """Shares = dollar risk / per-share risk, rounded down, and no more than the account can buy."""
    if risk <= 0:
        return 0
    return max(0, min(math.floor(risk_budget(cfg) / risk), math.floor(cfg.account / entry)))


def plan(side: int, entry: float, structure: float, swing: float | None, opposing: float | None,
         cfg: PriceActionConfig) -> Plan | None:
    """Stop just beyond the structure (or the LTF swing, if configured and further). Target is
    rr_target x risk or the next opposing level, whichever comes first. None if there's no room."""
    if side > 0:
        anchor = min(structure, swing) if cfg.stop_at_swing and swing is not None else structure
        stop = round(anchor - cfg.stop_buffer, 4)
    else:
        anchor = max(structure, swing) if cfg.stop_at_swing and swing is not None else structure
        stop = round(anchor + cfg.stop_buffer, 4)
    risk = side * (entry - stop)
    if risk <= 0:
        return None
    rr_px = entry + side * cfg.rr_target * risk
    target, kind = rr_px, "rr"
    if opposing is not None and side * (opposing - entry) < side * (rr_px - entry):
        target, kind = opposing, "level"
    if side * (target - entry) < cfg.min_room_r * risk:
        return None
    return Plan(side, round(entry, 4), stop, round(target, 4), round(risk, 4), size(entry, risk, cfg), kind)


def walk(p: Plan, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, start: int, last: int,
         cfg: PriceActionConfig) -> tuple[int, float, str]:
    """From bar `start` to `last` (the flatten bar). Within a bar the stop is checked before the
    target (the cautious assumption when one bar spans both). Break-even is armed once a bar reaches
    +1R and takes effect from the next bar. Returns (exit bar, exit price, reason)."""
    stop, be_done = p.stop, False
    for j in range(start, last + 1):
        if p.side > 0:
            if l[j] <= stop:
                return j, min(stop, o[j]), "breakeven" if be_done else "stop"
            if h[j] >= p.target:
                return j, max(p.target, o[j]), "target"
            if not be_done and h[j] >= p.entry + cfg.breakeven_at_r * p.risk:
                stop, be_done = max(stop, p.entry), True
        else:
            if h[j] >= stop:
                return j, max(stop, o[j]), "breakeven" if be_done else "stop"
            if l[j] <= p.target:
                return j, min(p.target, o[j]), "target"
            if not be_done and l[j] <= p.entry - cfg.breakeven_at_r * p.risk:
                stop, be_done = min(stop, p.entry), True
    return last, float(c[last]), "flatten"
