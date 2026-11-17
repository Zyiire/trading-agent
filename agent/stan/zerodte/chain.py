"""Option chain snapshots from free sources.

  Alpaca chain snapshot (free "indicative" feed): bid/ask, last trade, IV, greeks and the day's
      volume for every contract. Indicative quotes are delayed about 15 minutes and adjusted, so they
      are fine for levels and put/call, and not trusted for fill prices (Phase 3 models fills).
  Alpaca contracts list: open interest as of the previous close, for every strike and expiry.
  Yahoo option chain: volume, open interest, IV, bid/ask per strike. The backup when Alpaca fails.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

from agent.broker.base import parse_occ_symbol


@dataclass(frozen=True)
class ChainRow:
    expiry: date
    strike: float
    type: str  # "call" or "put"
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    volume: float | None = None
    oi: float | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None or self.ask <= 0 or self.bid < 0:
            return self.last
        return (self.bid + self.ask) / 2


def _f(x):
    try:
        x = None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None
    return None if x is None or x != x else x  # NaN (Yahoo's "no trades") becomes None


# ----- Alpaca -----
def parse_alpaca_raw(raw: dict) -> list[ChainRow]:
    """Alpaca's raw chain-snapshot JSON ({contract: snapshot}, possibly under "snapshots")."""
    snaps = raw.get("snapshots", raw) if isinstance(raw, dict) else {}
    rows = []
    for contract, s in snaps.items():
        try:
            _, expiry, kind, strike = parse_occ_symbol(contract)
        except (ValueError, TypeError):
            continue
        s = s or {}
        q, t, g, d = (s.get(k) or {} for k in ("latestQuote", "latestTrade", "greeks", "dailyBar"))
        rows.append(ChainRow(expiry, strike, kind, _f(q.get("bp")), _f(q.get("ap")), _f(t.get("p")),
                             _f(s.get("impliedVolatility")), _f(g.get("delta")), _f(g.get("gamma")),
                             _f(d.get("v"))))
    return rows


def alpaca_chain(raw_client, symbol: str, today: date, max_dte: int, lo: float, hi: float) -> list[ChainRow]:
    """`raw_client` is an OptionHistoricalDataClient made with raw_data=True (it keeps the daily volume)."""
    from alpaca.data.enums import OptionsFeed
    from alpaca.data.requests import OptionChainRequest

    req = OptionChainRequest(underlying_symbol=symbol, feed=OptionsFeed.INDICATIVE,
                             expiration_date_gte=today, expiration_date_lte=today + timedelta(days=max_dte),
                             strike_price_gte=round(lo, 2), strike_price_lte=round(hi, 2))
    return parse_alpaca_raw(raw_client.get_option_chain(req))


def alpaca_open_interest(trading_client, symbol: str, today: date, max_dte: int, lo: float, hi: float
                         ) -> list[tuple[date, float, str, float]]:
    """(expiry, strike, type, open interest) for every listed contract in range."""
    from alpaca.trading.requests import GetOptionContractsRequest

    out, token = [], None
    while True:
        req = GetOptionContractsRequest(underlying_symbols=[symbol], expiration_date_gte=today,
                                        expiration_date_lte=today + timedelta(days=max_dte),
                                        strike_price_gte=str(round(lo, 2)), strike_price_lte=str(round(hi, 2)),
                                        limit=10000, page_token=token)
        resp = trading_client.get_option_contracts(req)
        for c in getattr(resp, "option_contracts", None) or []:
            oi = _f(getattr(c, "open_interest", None))
            if oi is None:
                continue
            kind = str(getattr(c.type, "value", c.type)).lower()
            out.append((c.expiration_date, float(c.strike_price), kind, oi))
        token = getattr(resp, "next_page_token", None)
        if not token:
            return out


# ----- Yahoo -----
def yahoo_chain(symbol: str, today: date, max_dte: int, ticker=None) -> list[ChainRow]:
    import yfinance as yf

    tk = ticker or yf.Ticker(symbol)
    rows: list[ChainRow] = []
    for exp in tk.options:
        expiry = date.fromisoformat(exp)
        if not today <= expiry <= today + timedelta(days=max_dte):
            continue
        ch = tk.option_chain(exp)
        for kind, df in (("call", ch.calls), ("put", ch.puts)):
            rows += parse_yahoo_frame(df, expiry, kind)
    return rows


def parse_yahoo_frame(df, expiry: date, kind: str) -> list[ChainRow]:
    out = []
    for r in df.to_dict("records"):
        out.append(ChainRow(expiry, float(r["strike"]), kind, _f(r.get("bid")), _f(r.get("ask")),
                            _f(r.get("lastPrice")), _f(r.get("impliedVolatility")), volume=_f(r.get("volume")),
                            oi=_f(r.get("openInterest"))))
    return out


def with_open_interest(rows: list[ChainRow], oi: dict[tuple[str, float, str], float]) -> list[ChainRow]:
    """Attach stored open interest (keyed by expiry text, strike, type) to chain rows that lack it."""
    return [r if r.oi is not None else replace(r, oi=oi.get((r.expiry.isoformat(), r.strike, r.type)))
            for r in rows]


def rows_from_store(records: list[dict]) -> list[ChainRow]:
    return [ChainRow(date.fromisoformat(r["expiry"]), r["strike"], r["type"], r["bid"], r["ask"], r["last"],
                     r["iv"], r["delta"], r["gamma"], r["volume"]) for r in records]
