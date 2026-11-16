"""Price-action strategy: HTF market structure, key levels, confirmation candles at those levels,
and structural risk management. Raw OHLCV only, no oscillators, and nothing reads a bar that
hasn't closed.

    structure.py  swing pivots (confirmed N bars later) and HH/HL vs LH/LL trend
    levels.py     PDH/PDL, premarket high/low, recent swing levels, zone buffer
    patterns.py   pin bar, engulfing with volume, failed breakout
    risk.py       structural stop, position size, 2R-or-next-level target, break-even, the trade walk
    engine.py     signal at each closed bar, trades per session
    exam.py       Stan's stress tests for it
    config.py     every threshold, in PriceActionConfig
"""
from agent.stan.price_action.config import PriceActionConfig  # noqa: F401
from agent.stan.price_action.engine import context, run, signal_at, trades_for  # noqa: F401

__all__ = ["PriceActionConfig", "context", "run", "signal_at", "trades_for"]
