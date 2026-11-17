"""Settings for the 0DTE bot. Risk numbers follow Zy's spec; data numbers keep the free feeds light."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ZeroDteConfig:
    symbol: str = "QQQ"
    db_path: Path = Path("data/zerodte.db")
    bar_minutes: int = 3
    backfill_since: str = "2021-01-01"  # the event calendar starts in 2021, so tests start there too
    # Option chain snapshots
    snapshot_minutes: int = 5            # how often `watch` saves the chain during the session
    chain_max_dte: int = 7               # expiries saved: today through a week out (GEX uses them all)
    chain_strike_band: float = 0.06      # strikes within 6% of the price
    # Levels
    round_step: float = 10.0             # $10 round numbers (TZ's 680 call)
    minor_round_step: float = 5.0
    level_window: float = 0.03           # levels within 3% of the price
    level_merge: float = 0.001           # levels closer than 0.1% are one level
    top_oi_strikes: int = 3
    # Risk rules from the spec (used from Phase 4 on; here so they live in one place)
    risk_per_trade: float = 0.01         # 1% of the account
    max_risk_per_trade: float = 0.02     # never above 2%
    daily_loss_limit: float = 0.03       # stop for the day after losing 3%
    max_trades_per_day: int = 3
    flat_by: str = "15:45"               # always flat before expiry
