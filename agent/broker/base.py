"""Broker-neutral data shapes and the interface every broker adapter implements.

The rest of the agent only talks to this interface, so Alpaca can be swapped
for Robinhood (or anything else) later without touching the brain or the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, Sequence


@dataclass(frozen=True)
class AccountInfo:
    account_id: str
    status: str
    is_paper: bool
    equity: float
    cash: float
    buying_power: float
    options_buying_power: float | None
    options_level: int | None
    pattern_day_trader: bool
    trading_blocked: bool
    multiplier: str | None  # "1" means a cash account
    last_equity: float | None = None  # equity at the previous close


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    asset_class: str
    qty: float
    avg_entry_price: float
    market_value: float | None
    unrealized_pl: float | None


@dataclass(frozen=True)
class MarketClock:
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


@dataclass(frozen=True)
class DailyBar:
    symbol: str
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None


@dataclass(frozen=True)
class OptionQuote:
    contract: str  # OCC symbol, e.g. AAPL261218C00150000
    underlying: str
    expiration: date
    option_type: str  # "call" or "put"
    strike: float
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    last_price: float | None
    implied_volatility: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None or self.ask <= 0:
            return None
        return round((self.bid + self.ask) / 2, 4)

    @property
    def spread_pct(self) -> float | None:
        """Bid-ask spread as a share of the mid price (the liquidity filter uses this)."""
        mid = self.mid
        if mid is None or mid <= 0:
            return None
        return round((self.ask - self.bid) / mid, 4)


def parse_occ_symbol(contract: str) -> tuple[str, date, str, float]:
    """Split an OCC option symbol into (underlying, expiration, type, strike)."""
    tail = contract[-15:]
    root = contract[:-15].strip()
    if len(contract) < 16 or not root or tail[6] not in "CP" or not (tail[:6] + tail[7:]).isdigit():
        raise ValueError(f"Not an OCC option symbol: {contract!r}")
    expiration = date(2000 + int(tail[0:2]), int(tail[2:4]), int(tail[4:6]))
    option_type = "call" if tail[6] == "C" else "put"
    strike = int(tail[7:]) / 1000
    return root, expiration, option_type, strike


@dataclass(frozen=True)
class OrderInfo:
    client_order_id: str
    symbol: str
    side: str          # "buy" or "sell"
    qty: float
    status: str        # Alpaca status, e.g. "accepted", "filled", "canceled"
    filled_qty: float
    filled_avg_price: float | None
    filled_at: datetime | None
    submitted_at: datetime | None = None


class Broker(Protocol):
    """Reads plus paper-only market-on-open orders (used by agent.stan.paper)."""

    def get_account(self) -> AccountInfo: ...

    def get_positions(self) -> list[PositionInfo]: ...

    def get_clock(self) -> MarketClock: ...

    def get_daily_bars(self, symbols: Sequence[str], start: date, end: date | None = None) -> list[DailyBar]: ...

    def get_option_chain(
        self,
        underlying: str,
        expiration_from: date,
        expiration_to: date,
        strike_min: float | None = None,
        strike_max: float | None = None,
    ) -> list[OptionQuote]: ...

    def submit_open_order(self, symbol: str, qty: int, side: str, client_order_id: str) -> OrderInfo: ...

    def submit_fractional_order(self, symbol: str, qty: float, side: str, client_order_id: str) -> OrderInfo: ...

    def get_order(self, client_order_id: str) -> OrderInfo | None: ...

    def recent_orders(self, limit: int = 20) -> list[OrderInfo]: ...

    def cancel_open_orders(self) -> int: ...
