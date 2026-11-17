"""The read-only order list the Discord tools use."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from agent.broker.alpaca import AlpacaBroker
from agent.broker.base import OrderInfo
from agent.config import Settings


class FakeTrading:
    def __init__(self):
        self.requests = []

    def get_orders(self, req):
        self.requests.append(req)
        t = datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc)
        return [SimpleNamespace(client_order_id="stan-SPY-1-buy", symbol="SPY", side=SimpleNamespace(value="buy"),
                                qty="52.31", status=SimpleNamespace(value="filled"), filled_qty="52.31",
                                filled_avg_price="663.57", filled_at=t, submitted_at=t)]


def test_recent_orders_lists_all_statuses_newest_first():
    trading = FakeTrading()
    broker = AlpacaBroker(Settings(api_key="k", secret_key="s"), trading, object(), object())
    orders = broker.recent_orders(5)
    assert orders == [OrderInfo("stan-SPY-1-buy", "SPY", "buy", 52.31, "filled", 52.31, 663.57,
                                datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc),
                                datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc))]
    req = trading.requests[0]
    assert req.limit == 5 and str(getattr(req.status, "value", req.status)) == "all"


def test_order_info_still_builds_without_submitted_at():
    o = OrderInfo("id", "SPY", "buy", 1.0, "accepted", 0.0, None, None)
    assert o.submitted_at is None
