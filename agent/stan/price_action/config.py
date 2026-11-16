"""Every tunable threshold of the price-action strategy, in one place."""
from __future__ import annotations

from dataclasses import dataclass

BAR_MINUTES = 5          # the lower timeframe (LTF): 5-minute bars, 9:30-4:00 = 78 a session
SESSION_BARS = 78


@dataclass(frozen=True)
class PriceActionConfig:
    # 1. market structure on the higher timeframe (HTF)
    htf_minutes: int = 15          # HTF bar size; must be a multiple of BAR_MINUTES
    pivot_n: int = 3               # a pivot beats N HTF bars on each side; confirmed only after the N right bars close
    structure_pivots: int = 2      # compare the last 2 swing highs and the last 2 swing lows

    # 2. support and resistance
    atr_period: int = 14           # LTF ATR, for the zone buffer
    zone_atr: float = 0.25         # a bar within 0.25 ATR of a level counts as a test
    zone_pct: float | None = None  # or a fixed fraction of price (e.g. 0.001); overrides zone_atr when set
    swing_levels: int = 3          # how many recent confirmed HTF swing highs and lows count as levels
    use_premarket: bool = True     # premarket high/low, when the data has premarket bars

    # 3. confirmation candles
    pin_wick_body: float = 2.0     # rejection wick at least 2x the body
    pin_close_frac: float = 0.30   # pin closes in the top (long) or bottom (short) 30% of its range
    engulf_vol_mult: float = 1.0   # engulfing bar's volume above 1.0x its 20-bar average
    vol_period: int = 20
    patterns: tuple[str, ...] = ("pin", "engulf", "sweep")

    # 4. risk and execution
    stop_buffer: float = 0.02      # dollars beyond the structure low/high (2 cents)
    stop_at_swing: bool = False    # True: stop beyond the local LTF swing when it is further than the candle
    rr_target: float = 2.0         # 1:2, or the next opposing level if that comes first
    min_room_r: float = 1.0        # skip if the next opposing level is closer than 1R
    breakeven_at_r: float = 1.0    # stop to entry once price reaches +1R
    fill: str = "next_open"        # "next_open" (realistic) or "close" (fill at the signal bar's close)
    skip_open_minutes: int = 15    # no entries in the first 15 minutes
    flatten_minutes: int = 15      # flat 15 minutes before the close (3:45 PM)
    max_trades_per_day: int = 3
    account: float = 50.0
    risk_dollars: float | None = 5.0   # fixed dollar risk per trade...
    risk_pct: float = 0.10             # ...or this fraction of the account when risk_dollars is None

    def __post_init__(self) -> None:
        if self.htf_minutes % BAR_MINUTES:
            raise ValueError("htf_minutes must be a multiple of the 5-minute bar")
        if self.fill not in ("next_open", "close"):
            raise ValueError("fill must be 'next_open' or 'close'")

    @property
    def htf_bars(self) -> int:
        return self.htf_minutes // BAR_MINUTES

    @property
    def first_entry_slot(self) -> int:
        """Index of the first 5-minute bar allowed to trigger an entry: with 15 minutes skipped, the
        9:45-9:50 bar, so no signal candle comes from the opening window."""
        return self.skip_open_minutes // BAR_MINUTES

    @property
    def flatten_slot(self) -> int:
        """Index of the bar whose close flattens everything (the 3:40 bar closes at 3:45 PM)."""
        return SESSION_BARS - self.flatten_minutes // BAR_MINUTES - 1
