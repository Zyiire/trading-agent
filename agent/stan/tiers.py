"""Stan's three arena winners, traded on the paper account.

    low     SPY sized to 15% volatility at a reduced size (73% unless the last arena run said
            otherwise), the rest in cash. Sized at month ends.
    medium  SPY sized to 15% volatility at full size. Sized at month ends.
    high    TQQQ (a 3x Nasdaq fund) while QQQ closes above its 200-day average; otherwise cash.
            Checked every evening.

Each evening after the close, Stan works out the target weights from completed daily bars, compares
them with what the account holds, and sends fractional market orders that fill at the next open.
He only trades when a holding is more than 5% of the account away from its target, so small drifts
don't run up costs. The 20% kill switch sells everything and stops, exactly as before.

Stan manages the whole SPY and TQQQ positions in the account while a tier is on.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from agent.broker.base import Broker

TIER_SYMBOLS = ("SPY", "TQQQ")
SIGNAL = {"low": "SPY", "medium": "SPY", "high": "QQQ"}
DEFAULT_LOW_FRACTION = 0.73
TARGET_VOL = 0.15
BAND = 0.05


@dataclass(frozen=True)
class TierPlan:
    tier: str
    weights: dict[str, float]
    reason: str
    signal_day: str


def is_month_end(day: date) -> bool:
    return (pd.Timestamp(day) + pd.offsets.BDay(1)).month != day.month


def last_month_end_index(closes: pd.Series) -> int:
    """Position of the latest completed month-end bar (today counts if today ends the month)."""
    idx = closes.index
    if is_month_end(idx[-1].date()):
        return len(idx) - 1
    for k in range(len(idx) - 2, -1, -1):
        if idx[k].month != idx[k + 1].month:
            return k
    return len(idx) - 1


def plan(tier: str, history: dict[str, pd.DataFrame], low_fraction: float = DEFAULT_LOW_FRACTION) -> TierPlan:
    """Target weights from completed daily bars only."""
    if tier in ("low", "medium"):
        px = history["SPY"]["close"].astype(float).dropna()
        k = last_month_end_index(px)
        rv = float(px.pct_change().iloc[max(1, k - 20):k + 1].std() * np.sqrt(252))
        size = min(1.0, TARGET_VOL / rv) if rv > 0 else 1.0
        frac = low_fraction if tier == "low" else 1.0
        w = round(frac * size, 4)
        why = (f"last month's SPY volatility was {rv:.1%}, so {size:.0%} of the 15% target"
               + (f", times {frac:.0%} for low risk" if tier == "low" else "") + f": hold {w:.0%} SPY")
        return TierPlan(tier, {"SPY": w, "TQQQ": 0.0}, why, str(px.index[k].date()))
    if tier == "high":
        px = history["QQQ"]["close"].astype(float).dropna()
        avg = float(px.iloc[-200:].mean())
        on = len(px) >= 200 and float(px.iloc[-1]) > avg
        why = (f"QQQ closed at {px.iloc[-1]:.2f}, {'above' if on else 'below'} its 200-day average of {avg:.2f}: "
               + ("hold TQQQ" if on else "stay in cash"))
        return TierPlan(tier, {"SPY": 0.0, "TQQQ": 1.0 if on else 0.0}, why, str(px.index[-1].date()))
    raise ValueError(f"unknown tier {tier}")


def orders_for(p: TierPlan, equity: float, held: dict[str, float], prices: dict[str, float],
               band: float = BAND) -> list[tuple[str, str, float]]:
    """(symbol, side, qty) to move the holdings to the plan; sells first so cash is there for buys."""
    out = []
    for sym in TIER_SYMBOLS:
        px = prices.get(sym)
        if not px or px <= 0:
            continue
        have = held.get(sym, 0.0)
        cur_w = have * px / equity if equity > 0 else 0.0
        tgt_w = p.weights.get(sym, 0.0)
        if tgt_w == 0 and have > 0:
            out.append((sym, "sell", round(have, 6)))
        elif abs(tgt_w - cur_w) > band:
            qty = round(abs(tgt_w - cur_w) * equity / px, 6)
            side = "buy" if tgt_w > cur_w else "sell"
            if side == "sell":
                qty = min(qty, have)
            if qty > 0:
                out.append((sym, side, qty))
    return sorted(out, key=lambda o: o[1] != "sell")


def tier_order_id(sym: str, day: str, side: str) -> str:
    return f"stan-tier-{sym}-{day.replace('-', '')}-{side}"


def run_tier(broker: Broker, book, history: dict[str, pd.DataFrame], tier: str, day: str, equity: float,
             prices: dict[str, float], say, dry_run: bool = False, low_fraction: float = DEFAULT_LOW_FRACTION) -> None:
    """One evening for a tier: plan, compare, and send the orders (once per day)."""
    p = plan(tier, history, low_fraction)
    held = {pos.symbol: pos.qty for pos in broker.get_positions() if pos.symbol in TIER_SYMBOLS}
    say("thought", f"{tier.capitalize()}-risk plan: {p.reason}.")
    if book.get("tier_orders_day") == day:
        return
    todo = orders_for(p, equity, held, prices)
    if not todo:
        say("thought", "Holdings are within 5% of the plan, so no trade tonight.")
    for sym, side, qty in todo:
        oid = tier_order_id(sym, day, side)
        if dry_run:
            say("thought", f"Would {side} {qty:g} {sym} at the next open (about ${qty * prices[sym]:,.0f}).")
            continue
        try:
            broker.submit_fractional_order(sym, qty, side, oid)
        except Exception as err:
            try:
                taken = broker.get_order(oid) is not None
            except Exception:
                taken = False
            if not taken:
                say("fail", f"Couldn't send the {sym} {side} order: {type(err).__name__}: {err}")
                continue
        say("progress", f"{side.capitalize()} {qty:g} {sym} at the next open (about ${qty * prices[sym]:,.0f}).")
    if not dry_run:
        book.set("tier_orders_day", day)
        book.set("tier_plan", {"tier": tier, "weights": p.weights, "reason": p.reason, "day": day})


def sell_all(broker: Broker, day: str, say, dry_run: bool = False) -> None:
    for pos in broker.get_positions():
        if pos.symbol in TIER_SYMBOLS and pos.qty > 0:
            oid = tier_order_id(pos.symbol, day, "sell")
            if not dry_run:
                try:
                    broker.submit_fractional_order(pos.symbol, pos.qty, "sell", oid)
                except Exception as err:
                    say("fail", f"Couldn't send the {pos.symbol} sell: {err}")
                    continue
            say("fail", f"Selling all {pos.qty:g} {pos.symbol} at the next open.")


def now_et() -> datetime:
    from agent.stan.shadow import ET
    return datetime.now(ET)
