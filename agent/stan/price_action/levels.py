"""2. Key levels: previous day high/low, premarket high/low, recent confirmed HTF swings."""
from __future__ import annotations

from dataclasses import dataclass

from agent.stan.price_action.config import PriceActionConfig
from agent.stan.price_action.series import Series
from agent.stan.price_action.structure import Pivot, known_pivots


@dataclass(frozen=True)
class Level:
    name: str     # "PDH", "PDL", "PMH", "PML", "swing high", "swing low"
    price: float


def day_levels(s: Series, cfg: PriceActionConfig) -> dict[int, list[Level]]:
    """Per session: levels fixed before the open (previous session's high/low, today's premarket)."""
    out: dict[int, list[Level]] = {}
    for k, d in enumerate(s.days):
        lv = []
        if k > 0:
            prev = s.day_idx == k - 1
            lv += [Level("PDH", float(s.h[prev].max())), Level("PDL", float(s.l[prev].min()))]
        if cfg.use_premarket and d in s.premarket:
            ph, pl = s.premarket[d]
            lv += [Level("PMH", float(ph)), Level("PML", float(pl))]
        out[k] = lv
    return out


def levels_at(i: int, session_levels: list[Level], pivots: list[Pivot], cfg: PriceActionConfig) -> list[Level]:
    """Every level known at the close of LTF bar i."""
    lv = list(session_levels)
    lv += [Level("swing high", p) for p in known_pivots(pivots, i, "high", cfg.swing_levels)]
    lv += [Level("swing low", p) for p in known_pivots(pivots, i, "low", cfg.swing_levels)]
    return lv


def zone_buffer(price: float, atr_now: float, cfg: PriceActionConfig) -> float:
    return price * cfg.zone_pct if cfg.zone_pct else cfg.zone_atr * atr_now


def next_opposing(levels: list[Level], entry: float, side: int, buffer: float) -> float | None:
    """The nearest level beyond the entry in the trade's direction (resistance for longs)."""
    if side > 0:
        above = [lv.price for lv in levels if lv.price > entry + buffer]
        return min(above) if above else None
    below = [lv.price for lv in levels if lv.price < entry - buffer]
    return max(below) if below else None
