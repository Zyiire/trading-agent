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
    assert r["facts"] == ["On Nov 17 (today) Stan bought 52.31 SPY at $663.57 (filled).",
                          "On Nov 17 (today) Stan placed an order to sell 3 QQQ; status: accepted."]
    assert r["data"]["orders"][0]["day"] == "2026-11-17"


def test_recent_trades_labels_yesterday_and_leaves_older_days_plain(ctx):
    t = datetime(2026, 11, 16, 14, 35, tzinfo=timezone.utc)
    old = datetime(2026, 11, 10, 14, 35, tzinfo=timezone.utc)
    ctx.broker_factory = lambda: FakeBroker(orders=[
        OrderInfo("1", "SPY", "buy", 1.0, "filled", 1.0, 700.0, t, t),
        OrderInfo("2", "SPY", "buy", 1.0, "filled", 1.0, 700.0, old, old),
    ])
    facts = T.recent_trades(ctx)["facts"]
    assert facts[0].startswith("On Nov 16 (yesterday) Stan bought")
    assert facts[1].startswith("On Nov 10 Stan bought")


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
        {"name": "Stan Discord", "state": "Ready", "last_run": "2026-11-17T08:00:00", "last_result": 3221225786,
         "next_run": ""},
        {"name": "Stan odd", "state": "Ready", "last_run": "2026-11-17T08:00:00", "last_result": 3221225477,
         "next_run": ""},
        {"name": "Stan small", "state": "Ready", "last_run": "2026-11-17T08:00:00", "last_result": 1, "next_run": ""},
    ]
    facts = T.schedule(ctx)["facts"]
    assert "Stan open run: last ran Nov 17 at 9:35 AM (OK); next run Nov 18 at 9:35 AM." in facts
    assert "Stan nightly study: has never run; next run Nov 17 at 6:00 PM." in facts
    assert "Stan Discord: last ran Nov 17 at 8:00 AM (stopped: the window was closed or Ctrl+C); no next run scheduled." in facts
    assert "Stan odd: last ran Nov 17 at 8:00 AM (error code 0xC0000005); no next run scheduled." in facts
    assert "Stan small: last ran Nov 17 at 8:00 AM (error code 1); no next run scheduled." in facts
    assert "The trading run log was last written on Nov 17, exit code 0 (OK)." in facts
    assert "The study log was last written on Nov 16, but that run never finished (no exit code)." in facts
    assert "The data collector log has no entries yet." in facts


def test_schedule_with_no_tasks(ctx):
    assert "No Stan tasks are scheduled in Windows." in T.schedule(ctx)["facts"]


from agent.stan.journal import Journal


def seed_journal(path, state="finished", mode="real"):
    j = Journal(path)
    j.set_meta("run_id", "20261116-231341-818600")
    j.set_meta("state", state)
    j.set_meta("mode", mode)
    j.set_meta("summary", {"stan_growth": 1.12, "hold_growth": 6.29, "trades": 8818, "win_rate": 0.575,
                           "years_tested": 11, "years_passed": 9, "years_beat_hold": 2, "max_dd": -0.1784,
                           "hold_max_dd": -0.3514, "ready_for_paper": False})
    j.set_meta("arena", {"winners": {"low": "vt_s", "medium": "vt", "high": "lev"}, "strategies": [
        {"key": "vt_s", "name": "Volatility target at 73% size", "cagr": 0.0972, "max_dd": -0.1844},
        {"key": "vt", "name": "Volatility target", "cagr": 0.1248, "max_dd": -0.2459},
        {"key": "lev", "name": "3x QQQ with a 200-day trend switch", "cagr": 0.3131, "max_dd": -0.6128}]})
    j.record_exam({"year": 2016, "n_calls": 100, "accuracy": 0.52, "baseline_up": 0.53, "trades": 753,
                   "win_rate": 0.5976, "profit_factor": 1.26, "stan_return": 0.0382, "hold_return": 0.2513,
                   "passed": True, "beat_hold": False})
    j.add_lesson("mistake", "I do worse when the stock is overbought", "Across 373 trades I won 53.3%.")
    j.add_lesson("strength", "I do best after a sharp selloff", "Across 291 trades I won 69.4%.")
    j.close()


def test_study_report_facts(ctx):
    seed_journal(ctx.journal_path)
    facts = T.study_report(ctx)["facts"]
    assert facts[0] == "Last study term: Nov 16."
    assert "Verdict: not ready yet, because simply holding beat him." in facts
    assert "$1 with Stan became $1.12. $1 simply held became $6.29. Holding did better by $5.17." in facts
    assert "He won 57.5% of 8,818 practice trades." in facts
    assert "He passed 9 of 11 yearly exams and beat simply holding in 2 of them." in facts
    assert "His worst drop was -17.8%; simply holding's worst drop was -35.1%." in facts
    assert "Real money stays locked until the go-live checks pass." in facts


def test_study_report_while_in_class_and_in_demo_mode(ctx):
    seed_journal(ctx.journal_path, state="training", mode="demo")
    facts = T.study_report(ctx)["facts"]
    assert "He is in class right now, so these are the results of the last finished term." in facts
    assert "These results come from made-up practice data, not real market history." in facts


def test_study_report_before_any_term(ctx):
    assert T.study_report(ctx)["facts"] == ["Stan hasn't finished a study term yet."]


def test_exams_and_lessons(ctx):
    seed_journal(ctx.journal_path)
    assert T.exams(ctx)["facts"] == [
        "2016: passed. Stan +3.8% vs simply holding +25.1%; 753 trades, won 59.8%."]
    assert T.lessons(ctx)["facts"] == [
        "Strength: I do best after a sharp selloff. Across 291 trades I won 69.4%.",
        "Mistake: I do worse when the stock is overbought. Across 373 trades I won 53.3%."]
    assert len(T.lessons(ctx, limit=1)["facts"]) == 1


def test_strategy_facts(ctx):
    seed_journal(ctx.journal_path)
    book = PaperBook(ctx.journal_path)
    book.set("tier_plan", {"tier": "medium", "reason": "hold 100% SPY", "day": "2026-11-13"})
    book.close()
    facts = T.strategy(ctx)["facts"]
    assert facts[0] == "The paper account trades the medium-risk tier."
    assert "Best medium-risk strategy: Volatility target, 12.5% a year, worst drop -24.6%." in facts
    assert "Best high-risk strategy: 3x QQQ with a 200-day trend switch, 31.3% a year, worst drop -61.3%." in facts
    assert "Latest plan (Nov 13): hold 100% SPY." in facts


import ast

from agent.discord_bot import brain as Bmod


def test_levels_without_any_snapshot(ctx):
    assert T.levels(ctx)["facts"] == ["No QQQ options snapshot is saved for Nov 17."]


def test_levels_uses_the_latest_saved_day(ctx, monkeypatch):
    from agent.stan.zerodte import collect

    seen = []

    def fake_view(store, cfg, day):
        seen.append(day)
        return {"day": day} if day == date(2026, 11, 16) else None

    monkeypatch.setattr(collect, "today_view", fake_view)
    monkeypatch.setattr(collect, "report", lambda view, day: "QQQ 661.63 at 2026-11-16 23:01 ET\n\nLevels:\n  660.00  below")
    monkeypatch.setattr("agent.stan.zerodte.store.ZeroStore.snapshot_days", lambda self, sym: {"2026-11-16": 1})
    r = T.levels(ctx)
    assert seen == [date(2026, 11, 17), date(2026, 11, 16)]
    assert r["facts"] == ["QQQ 661.63 at 2026-11-16 23:01 ET", "Levels:", "660.00  below"]


def test_registry_specs_and_errors(ctx):
    names = [s["function"]["name"] for s in T.tool_specs()]
    assert names == ["account", "recent_trades", "trading_log", "study_report", "exams", "lessons", "strategy",
                     "box_breakout", "levels", "schedule", "market_clock"]
    assert len(names) == 11
    desc = next(s for s in T.tool_specs() if s["function"]["name"] == "box_breakout")["function"]["description"]
    assert all(w in desc for w in ("box", "breakout", "opening range", "QQQ trade today", "box backtest", "how did the box do"))
    assert "school" in next(s for s in T.tool_specs() if s["function"]["name"] == "study_report")["function"]["description"]
    assert T.run_tool(ctx, "place_order", {}) == {"error": "There is no tool called place_order."}
    assert T.run_tool(ctx, "trading_log", {"limit": 1, "symbol": "SPY"})["facts"] == ["Nov 17: Buy 52.31 SPY (about $34,700)."]
    ctx.broker_factory = lambda: (_ for _ in ()).throw(ConnectionError("Alpaca down"))
    assert T.run_tool(ctx, "account", {}) == {"error": "ConnectionError: Alpaca down"}


FORBIDDEN = {"submit_open_order", "submit_fractional_order", "cancel_open_orders", "safe_submit", "run_evening",
             "run_tier", "sell_all", "set_strategy", "run_training", "record_equity", "add_trade", "set_meta",
             "think", "set", "Popen", "train"}


def test_discord_package_is_read_only():
    pkg = Path(T.__file__).parent
    for src in pkg.glob("*.py"):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN, f"{src.name} uses .{node.attr}"
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("agent.stan.train", "agent.stan.tiers"), f"{src.name} imports {node.module}"
                assert not {a.name for a in node.names} & FORBIDDEN, f"{src.name} imports a write function"


@pytest.mark.parametrize("bad", [[1], "5", 5, "[1]"])
def test_run_tool_tolerates_non_dict_arguments(ctx, bad):
    assert "facts" in T.run_tool(ctx, "trading_log", bad)


# ---------------------------------------------------------------- box breakout
BOX_EXAM = {"span": [2016, 2026], "headline": {"trades": 1327, "win_rate": 0.362, "avg_r": 0.139},
            "facts": ["I replayed the video's box breakout on QQQ.", "So the rules still paid on QQQ."]}


def box_ctx(ctx, state=None, results=None, exam=None, strategy="box"):
    book = PaperBook(ctx.journal_path)
    book.set("strategy", strategy)
    if state is not None:
        book.set("box_today", state)
    if results is not None:
        book.set("box_results", results)
    book.log("2026-11-17", "thought", "Box breakout: no trade today, nothing closed outside the box.")
    book.close()
    if exam is not None:
        j = Journal(ctx.journal_path)
        j.set_meta("box_exam", exam)
        j.close()
    return ctx


def test_box_breakout_without_any_data(ctx):
    box_ctx(ctx, strategy="medium")
    facts = T.box_breakout(ctx)["facts"]
    assert "No box trade has been recorded yet." in facts
    assert any("hasn't run the box backtest" in f for f in facts)
    assert any(f.startswith("Note from Nov 17 (today): Box breakout") for f in facts)
    assert any("trades medium" in f or "medium-risk tier" in f for f in facts)   # not on the box right now


def test_box_breakout_exam_facts_present(ctx):
    box_ctx(ctx, exam=BOX_EXAM)
    facts = T.box_breakout(ctx)["facts"]
    assert "I replayed the video's box breakout on QQQ." in facts
    assert not any("hasn't run the box backtest" in f for f in facts)


def test_box_breakout_sent_closed_and_skipped_states(ctx):
    order = {"id": "stan-box-20261117", "qty": 120, "side": "buy", "stop": 650.0, "target": 665.0, "risk": 5.0}
    box_ctx(ctx, {"day": "2026-11-17", "status": "sent", "reason": "r", "box": [655.0, 650.0], "order": order})
    f = T.box_breakout(ctx)["facts"]
    assert any(x.startswith("On Nov 17 (today) Stan bought 120 QQQ") and "stop $650.00" in x and "target $665.00" in x
               and "The box was $655.00 high to $650.00 low." in x for x in f)
    done = {"day": "2026-11-16", "side": "buy", "qty": 120, "entry": 655.5, "exit": 650.0, "r": -1.1, "pnl": -660.0,
            "reason": "stop"}
    box_ctx(ctx, {"day": "2026-11-16", "status": "closed", "order": order, "result": done}, [done])
    f = T.box_breakout(ctx)["facts"]
    assert any("On Nov 16 (yesterday) the box trade was long 120 QQQ from $655.50 to $650.00 (hit the stop)" in x
               and "-1.10R" in x for x in f)
    assert "Closed box trades so far: 1, totaling -1.10R." in f
    box_ctx(ctx, {"day": "2026-11-17", "status": "skipped", "reason": "half day"})
    assert "On Nov 17 (today) Stan made no box trade: half day." in T.box_breakout(ctx)["facts"]
    box_ctx(ctx, {"day": "2026-11-17", "status": "no_box", "reason": "the box is incomplete"})
    assert any("no box trade: the box is incomplete" in x for x in T.box_breakout(ctx)["facts"])


def test_strategy_names_box_and_adds_the_exam_headline(ctx):
    box_ctx(ctx, exam=BOX_EXAM)
    facts = T.strategy(ctx)["facts"]
    assert facts[0] == "The paper account trades the box breakout on QQQ."
    assert facts[1] == ("Box breakout backtest on QQQ from 2016 to 2026: 1,327 trades, won 36.2%, "
                        "averaging +0.139R a trade.")
    box_ctx(ctx, exam=BOX_EXAM, strategy="medium")
    assert not any("Box breakout backtest" in f for f in T.strategy(ctx)["facts"])


def test_box_breakout_more_states_and_exit_words(ctx):
    base = {"day": "2026-11-17"}
    box_ctx(ctx, {**base, "status": "flatten_failed", "reason": "no fill",
                  "order": {"qty": 5, "side": "sell", "stop": 660.0, "target": 640.0}})
    f = T.box_breakout(ctx)["facts"]
    assert any("Stan shorted 5 QQQ" in x and "The 3:55 close ran into a problem: no fill." in x for x in f)
    box_ctx(ctx, {**base, "status": "closed", "result": {"side": "buy", "qty": 0, "entry": None}})
    assert "On Nov 17 (today) the box order never filled, so nothing was traded." in T.box_breakout(ctx)["facts"]
    box_ctx(ctx, {**base, "status": "failed", "reason": "the order failed"})
    assert "On Nov 17 (today) Stan made no box trade: the order failed." in T.box_breakout(ctx)["facts"]
    box_ctx(ctx, {**base, "status": "blocked", "reason": "the trend is down"})
    assert "On Nov 17 (today) Stan made no box trade: the trend is down." in T.box_breakout(ctx)["facts"]
    for why, words in (("target", "hit the 3R target"), ("time", "closed at 3:55"), ("early", "closed the next morning")):
        res = {"side": "buy", "qty": 1, "entry": 1.0, "exit": 2.0, "r": 1.0, "pnl": 1.0, "reason": why}
        box_ctx(ctx, {**base, "status": "closed", "result": res})
        assert any(f"({words})" in x for x in T.box_breakout(ctx)["facts"])
