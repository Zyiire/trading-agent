"""Alpaca adapter: account, positions, clock, bars, option chains, and
paper-only market-on-open orders for Stan's paper trading.

Live trading is impossible here: the constructor refuses non-paper settings
and the trading client is always created with paper=True.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Sequence

from alpaca.data.enums import Adjustment, DataFeed, OptionsFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrderByIdRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from agent.broker.base import (
    AccountInfo,
    DailyBar,
    MarketClock,
    OptionQuote,
    OrderInfo,
    PositionInfo,
    parse_occ_symbol,
)
from agent.config import Settings


def _f(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _enum_text(value) -> str:
    return str(getattr(value, "value", value))


def _not_found(err: Exception) -> bool:
    return "404" in str(err) or "not found" in str(err).lower() or "does not exist" in str(err).lower()


# Order statuses that can't change any more; anything else may still fill, so it can be cancelled.
FINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced", "pending_cancel", "done_for_day",
                  "stopped", "suspended"}


class AlpacaBroker:
    def __init__(
        self,
        settings: Settings,
        trading_client=None,
        stock_data_client=None,
        option_data_client=None,
    ) -> None:
        if not settings.paper:
            raise RuntimeError("Live trading is locked in this phase.")
        self._paper = settings.paper
        self._trading = trading_client or TradingClient(
            settings.api_key, settings.secret_key, paper=True
        )
        self._stocks = stock_data_client or StockHistoricalDataClient(
            settings.api_key, settings.secret_key
        )
        self._options = option_data_client or OptionHistoricalDataClient(
            settings.api_key, settings.secret_key
        )

    # ----- account -----
    def get_account(self) -> AccountInfo:
        a = self._trading.get_account()
        level = getattr(a, "options_trading_level", None)
        return AccountInfo(
            account_id=str(a.id),
            status=_enum_text(a.status),
            is_paper=self._paper,
            equity=_f(a.equity) or 0.0,
            cash=_f(a.cash) or 0.0,
            buying_power=_f(a.buying_power) or 0.0,
            options_buying_power=_f(getattr(a, "options_buying_power", None)),
            options_level=int(level) if level is not None else None,
            pattern_day_trader=bool(a.pattern_day_trader),
            trading_blocked=bool(a.trading_blocked),
            multiplier=str(a.multiplier) if a.multiplier is not None else None,
            last_equity=_f(getattr(a, "last_equity", None)),
        )

    def get_positions(self) -> list[PositionInfo]:
        return [
            PositionInfo(
                symbol=p.symbol,
                asset_class=_enum_text(p.asset_class),
                qty=_f(p.qty) or 0.0,
                avg_entry_price=_f(p.avg_entry_price) or 0.0,
                market_value=_f(p.market_value),
                unrealized_pl=_f(p.unrealized_pl),
            )
            for p in self._trading.get_all_positions()
        ]

    def get_clock(self) -> MarketClock:
        c = self._trading.get_clock()
        return MarketClock(c.timestamp, bool(c.is_open), c.next_open, c.next_close)

    # ----- market data -----
    def get_daily_bars(
        self, symbols: Sequence[str], start: date, end: date | None = None
    ) -> list[DailyBar]:
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, time.min, tzinfo=timezone.utc),
            end=datetime.combine(end, time.max, tzinfo=timezone.utc) if end else None,
            adjustment=Adjustment.ALL,  # split and dividend adjusted
            feed=DataFeed.IEX,  # the free feed
        )
        barset = self._stocks.get_stock_bars(request)
        bars: list[DailyBar] = []
        for symbol, rows in barset.data.items():
            for b in rows:
                bars.append(
                    DailyBar(
                        symbol=symbol,
                        day=b.timestamp.date(),
                        open=float(b.open),
                        high=float(b.high),
                        low=float(b.low),
                        close=float(b.close),
                        volume=float(b.volume),
                        vwap=_f(b.vwap),
                    )
                )
        return bars

    def get_option_chain(
        self,
        underlying: str,
        expiration_from: date,
        expiration_to: date,
        strike_min: float | None = None,
        strike_max: float | None = None,
    ) -> list[OptionQuote]:
        request = OptionChainRequest(
            underlying_symbol=underlying,
            feed=OptionsFeed.INDICATIVE,  # free feed; quotes are delayed about 15 minutes
            expiration_date_gte=expiration_from,
            expiration_date_lte=expiration_to,
            strike_price_gte=strike_min,
            strike_price_lte=strike_max,
        )
        chain = self._options.get_option_chain(request)
        quotes: list[OptionQuote] = []
        for contract, snap in chain.items():
            try:
                _, expiration, option_type, strike = parse_occ_symbol(contract)
            except ValueError:
                continue
            q = snap.latest_quote
            t = snap.latest_trade
            g = snap.greeks
            quotes.append(
                OptionQuote(
                    contract=contract,
                    underlying=underlying,
                    expiration=expiration,
                    option_type=option_type,
                    strike=strike,
                    bid=_f(q.bid_price) if q else None,
                    ask=_f(q.ask_price) if q else None,
                    bid_size=_f(q.bid_size) if q else None,
                    ask_size=_f(q.ask_size) if q else None,
                    last_price=_f(t.price) if t else None,
                    implied_volatility=_f(snap.implied_volatility),
                    delta=_f(g.delta) if g else None,
                    gamma=_f(g.gamma) if g else None,
                    theta=_f(g.theta) if g else None,
                    vega=_f(g.vega) if g else None,
                )
            )
        return quotes


    # ----- paper orders (Stan's paper trading) -----
    @staticmethod
    def _order_info(o) -> OrderInfo:
        return OrderInfo(
            client_order_id=str(o.client_order_id),
            symbol=o.symbol,
            side=_enum_text(o.side),
            qty=_f(o.qty) or 0.0,
            status=_enum_text(o.status),
            filled_qty=_f(o.filled_qty) or 0.0,
            filled_avg_price=_f(o.filled_avg_price),
            filled_at=o.filled_at,
            submitted_at=getattr(o, "submitted_at", None),
            order_type=_enum_text(getattr(o, "order_type", None)) if getattr(o, "order_type", None) else None,
            limit_price=_f(getattr(o, "limit_price", None)),
            stop_price=_f(getattr(o, "stop_price", None)),
        )

    def submit_open_order(self, symbol: str, qty: int, side: str, client_order_id: str) -> OrderInfo:
        """A market order that fills at the next opening auction (time in force OPG)."""
        if not self._paper:
            raise RuntimeError("Live trading is locked.")
        if qty <= 0 or int(qty) != qty:
            raise ValueError("Opening-auction orders need a whole number of shares.")
        order = self._trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=int(qty), side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.OPG, client_order_id=client_order_id,
        ))
        return self._order_info(order)

    def submit_fractional_order(self, symbol: str, qty: float, side: str, client_order_id: str) -> OrderInfo:
        """A market order for a fractional quantity. Alpaca allows fractions only as day orders; sent
        after the close, it waits and fills when the market opens."""
        if not self._paper:
            raise RuntimeError("Live trading is locked.")
        qty = round(float(qty), 6)
        if qty <= 0:
            raise ValueError("Quantity must be positive.")
        order = self._trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, client_order_id=client_order_id,
        ))
        return self._order_info(order)

    def get_order(self, client_order_id: str) -> OrderInfo | None:
        try:
            return self._order_info(self._trading.get_order_by_client_id(client_order_id))
        except Exception as err:  # Alpaca answers 404 for an unknown client id
            if _not_found(err):
                return None
            raise

    # ----- box breakout (agent.stan.box.live) -----
    def submit_bracket_order(self, symbol: str, qty: int, side: str, stop: float, target: float,
                             client_order_id: str) -> OrderInfo:
        """A market entry with a stop-loss and a take-profit attached (a day order, whole shares).
        `side` "buy" opens a long, "sell" a short. Prices are rounded to cents."""
        if not self._paper:
            raise RuntimeError("Live trading is locked.")
        if qty <= 0 or int(qty) != qty:
            raise ValueError("Bracket orders need a whole number of shares.")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy or sell, not {side!r}")
        order = self._trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=int(qty), side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET, client_order_id=client_order_id,
            take_profit=TakeProfitRequest(limit_price=round(float(target), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop), 2)),
        ))
        return self._order_info(order)

    def get_bracket(self, client_order_id: str) -> tuple[OrderInfo | None, list[OrderInfo]]:
        """Read-only: a bracket's entry order and its two exit legs. (None, []) if Alpaca has no such id."""
        try:
            parent = self._trading.get_order_by_client_id(client_order_id)
        except Exception as err:
            if _not_found(err):
                return None, []
            raise
        full = self._trading.get_order_by_id(parent.id, filter=GetOrderByIdRequest(nested=True))
        return self._order_info(full), [self._order_info(leg) for leg in (getattr(full, "legs", None) or [])]

    def cancel_orders(self, prefix: str) -> int:
        """Cancel every unfinished order whose client id starts with `prefix`, and the legs under it.
        Looks at the 100 newest orders, so a filled entry with open legs is found too."""
        if not self._paper:
            raise RuntimeError("Live trading is locked.")
        if not prefix.startswith("stan-"):
            raise ValueError("Only Stan's own orders (client ids starting 'stan-') can be cancelled.")
        orders = self._trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True, limit=100))
        count = 0
        for o in orders:
            if not str(o.client_order_id).startswith(prefix):
                continue
            for x in [o, *(getattr(o, "legs", None) or [])]:
                if _enum_text(x.status) in FINAL_STATUSES:
                    continue
                try:
                    self._trading.cancel_order_by_id(x.id)
                    count += 1
                except Exception:   # it filled or finished a moment ago; the caller re-reads the fills
                    pass
        return count

    def close_position(self, symbol: str, qty: float, side: str, client_order_id: str) -> OrderInfo:
        """Close `qty` shares of one of Stan's trades: a day market order on `side` ("sell" closes a long,
        "buy" covers a short), with Stan's own client id so it can be found and cancelled.

        The side comes from the caller's trade, not from the account's net position: another of Stan's
        trades may hold the same symbol the other way."""
        if not self._paper:
            raise RuntimeError("Live trading is locked.")
        if qty <= 0 or int(qty) != qty:
            raise ValueError("Closing needs a whole number of shares.")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy or sell, not {side!r}")
        order = self._trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=int(qty), side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, client_order_id=client_order_id,
        ))
        return self._order_info(order)

    def recent_orders(self, limit: int = 20) -> list[OrderInfo]:
        """Read-only: the account's latest orders of any status, newest first."""
        orders = self._trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=int(limit)))
        return [self._order_info(o) for o in orders]

    def cancel_open_orders(self) -> int:
        open_orders = self._trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        count = 0
        for o in open_orders:
            if str(o.client_order_id).startswith("stan-"):
                self._trading.cancel_order_by_id(o.id)
                count += 1
        return count
