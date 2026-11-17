"""Stan's read-only Discord tools, against a temporary journal and a fake broker."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from agent.broker.base import AccountInfo, MarketClock, OrderInfo, PositionInfo
from agent.discord_bot import tools as T
from agent.stan.paper import PaperBook

NOW = datetime(2026, 11, 17, 10, 0, tzinfo=T.ET)


class FakeBroker:
    def __init__(self, orders=None, open_=True):
        self.orders = orders if orders is not None else []
        self.open_ = open_

    def get_account(self):
        return AccountInfo("a", "ACTIVE", True, 100412.37, 61250.10, 61250.10, None, 3, False, False, "1",
                           last_equity=100000.0)

    def get_positions(self):
        return [PositionInfo("SPY", "us_equity", 52.31, 650.0, 34711.40, 100.0),
                PositionInfo("QQQ", "us_equity", 6.62, 660.0, 4450.87, 20.0)]

    def get_clock(self):
        return MarketClock(NOW, self.open_, datetime(2026, 11, 18, 14, 30, tzinfo=timezone.utc),
                           datetime(2026, 11, 17, 21, 0, tzinfo=timezone.utc))

    def recent_orders(self, limit=20):
        return self.orders[:limit]


@pytest.fixture
def ctx(tmp_path):
    journal = tmp_path / "stan.db"
    book = PaperBook(journal)
    book.set("strategy", "medium")
    book.log("2026-11-16", "thought", "Holdings are within 5% of the plan, so no trade is needed.")
    book.log("2026-11-17", "progress", "Buy 52.31 SPY (about $34,700).")
    book.close()
    return T.ToolContext(journal_path=str(journal), zerodte_db=str(tmp_path / "zerodte.db"), data_dir=tmp_path,
                         broker_factory=FakeBroker, task_reader=lambda: [], now=lambda: NOW)


def test_formatting_helpers():
    assert T.money(61250.1) == "$61,250.10" and T.money(-5) == "-$5.00" and T.money(None) == "unknown"
    assert T.nice_day(date(2026, 11, 7)) == "Nov 7"
    assert T.clock_time(datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc)) == "9:35 AM"
    assert T.result(["a"]) == {"facts": ["a"], "data": {}}


def test_account_facts(ctx):
    r = T.account(ctx)
    facts = " ".join(r["facts"])
    assert "Equity is $100,412.37." in facts and "Cash is $61,250.10." in facts
    assert "Today the account is up 0.41%." in facts
    assert "52.31 SPY worth $34,711.40" in facts and "6.62 QQQ worth $4,450.87" in facts
    assert "the medium-risk tier" in facts and "Trading is on (not paused)." in facts and "paper money" in facts
    assert r["data"]["strategy"] == "medium" and r["data"]["positions"][0]["symbol"] == "SPY"


def test_account_shows_a_pause_with_its_reason(ctx):
    book = PaperBook(ctx.journal_path)
    book.set("halted", True)
    book.set("halt_reason", "Paused by Zy")
    book.close()
    assert "Trading is paused (Paused by Zy)." in T.account(ctx)["facts"]


def test_recent_trades_filled_and_waiting(ctx):
    t = datetime(2026, 11, 17, 14, 35, tzinfo=timezone.utc)
    ctx.broker_factory = lambda: FakeBroker(orders=[
        OrderInfo("1", "SPY", "buy", 52.31, "filled", 52.31, 663.57, t, t),
        OrderInfo("2", "QQQ", "sell", 3.0, "accepted", 0.0, None, None, t),
    ])
    r = T.recent_trades(ctx)
    assert r["facts"] == ["On Nov 17 Stan bought 52.31 SPY at $663.57 (filled).",
                          "On Nov 17 Stan placed an order to sell 3 QQQ; status: accepted."]
    assert r["data"]["orders"][0]["day"] == "2026-11-17"


def test_recent_trades_when_there_are_none(ctx):
    assert T.recent_trades(ctx)["facts"] == ["No orders yet."]


def test_trading_log_newest_first(ctx):
    r = T.trading_log(ctx)
    assert r["facts"][0] == "Nov 17: Buy 52.31 SPY (about $34,700)."
    assert r["data"]["notes"][1]["day"] == "2026-11-16"


def test_market_clock(ctx):
    assert T.market_clock(ctx)["facts"] == ["The market is open. It closes at 4:00 PM New York time."]
    ctx.broker_factory = lambda: FakeBroker(open_=False)
    assert T.market_clock(ctx)["facts"] == ["The market is closed. It next opens Nov 18 at 9:30 AM New York time."]


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_last_log_entry_finished_unfinished_missing(tmp_path):
    p = write(tmp_path / "open.log", "==== Mon 11/16/2026  9:35:00.54 ====\nold\nExit code: 1\n"
                                     "==== Tue 11/17/2026  9:35:01.00 ====\nUsing: python\nExit code: 0\nSent the open report\n")
    e = T.last_log_entry(p)
    assert e.day == date(2026, 11, 17) and e.exit_code == 0 and e.lines[0].startswith("==== Tue")
    cut = write(tmp_path / "study.log", "==== Tue 11/17/2026 18:00:00.10 ====\nstudying...\n")
    assert T.last_log_entry(cut).exit_code is None
    assert T.last_log_entry(tmp_path / "missing.log") is None
    assert T.last_log_entry(write(tmp_path / "empty.log", "")) is None


def test_schedule_reports_tasks_and_logs(ctx, tmp_path):
    write(tmp_path / "open.log", "==== Tue 11/17/2026  9:35:01.00 ====\nExit code: 0\n")
    write(tmp_path / "study.log", "==== Mon 11/16/2026 18:00:00.10 ====\nstudying...\n")
    ctx.task_reader = lambda: [
        {"name": "Stan open run", "state": "Ready", "last_run": "2026-11-17T09:35:00", "last_result": 0,
         "next_run": "2026-11-18T09:35:00"},
        {"name": "Stan nightly study", "state": "Ready", "last_run": "1999-11-30T00:00:00", "last_result": 267011,
         "next_run": "2026-11-17T18:00:00"},
    ]
    facts = T.schedule(ctx)["facts"]
    assert "Stan open run: last ran Nov 17 at 9:35 AM (OK); next run Nov 18 at 9:35 AM." in facts
    assert "Stan nightly study: has never run; next run Nov 17 at 6:00 PM." in facts
    assert "The trading run log was last written on Nov 17, exit code 0 (OK)." in facts
    assert "The study log was last written on Nov 16, but that run never finished (no exit code)." in facts
    assert "The data collector log has no entries yet." in facts


def test_schedule_with_no_tasks(ctx):
    assert "No Stan tasks are scheduled in Windows." in T.schedule(ctx)["facts"]
