"""Fixed-wording reports, message splitting and the missed-run check."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import httpx
import pytest

from agent.discord_bot import notify as N
from agent.discord_bot import reports as R
from agent.discord_bot import tools as T

ROOT = Path(__file__).resolve().parents[1]

NOW = datetime(2026, 11, 17, 9, 40, tzinfo=T.ET)


def fake_tools(monkeypatch, answers):
    monkeypatch.setattr(R, "run_tool", lambda ctx, name, args: answers[name])


@pytest.fixture
def ctx(tmp_path):
    return T.ToolContext(data_dir=tmp_path, now=lambda: NOW)


def test_split_message_on_lines_and_hard_cuts():
    assert R.split_message("") == []
    assert R.split_message("short") == ["short"]
    parts = R.split_message("a" * 1500 + "\n" + "b" * 1500)
    assert parts == ["a" * 1500, "b" * 1500]
    long = R.split_message("x" * 4500)
    assert [len(p) for p in long] == [2000, 2000, 500]
    assert all(len(p) <= 2000 for p in R.split_message(("word " * 900).strip()))


ACCOUNT = {"facts": [], "data": {"equity": 100412.37, "day_change": 0.0041, "halted": False}}


def test_open_report_with_orders_today(ctx, monkeypatch):
    fake_tools(monkeypatch, {
        "recent_trades": {"facts": [], "data": {"orders": [
            {"day": "2026-11-17", "fact": "On Nov 17 (today) Stan bought 52.31 SPY at $663.57 (filled)."},
            {"day": "2026-11-13", "fact": "On Nov 13 Stan bought 1 QQQ at $600.00 (filled)."}]}},
        "trading_log": {"facts": [], "data": {"notes": []}},
        "account": ACCOUNT})
    assert R.job_report(ctx, "open", 0) == (
        "Trading run for Nov 17:\n"
        "On Nov 17 Stan bought 52.31 SPY at $663.57 (filled).\n"
        "Equity is $100,412.37, up 0.41% today. Paper money only.")


def test_open_report_without_orders_uses_todays_notes(ctx, monkeypatch):
    fake_tools(monkeypatch, {
        "recent_trades": {"facts": ["No orders yet."], "data": {"orders": []}},
        "trading_log": {"facts": [], "data": {"notes": [
            {"day": "2026-11-17", "text": "Holdings are within 5% of the plan, so no trade is needed."},
            {"day": "2026-11-16", "text": "old note"}]}},
        "account": {"facts": [], "data": {"equity": 99000.0, "day_change": -0.01, "halted": True}}})
    assert R.job_report(ctx, "open", 0) == (
        "Trading run for Nov 17:\n"
        "No orders today. Holdings are within 5% of the plan, so no trade is needed.\n"
        "Trading is paused, so no orders go out until it's switched back on in the web app.\n"
        "Equity is $99,000.00, down 1.00% today. Paper money only.")


def test_open_report_when_alpaca_is_down(ctx, monkeypatch):
    fake_tools(monkeypatch, {"recent_trades": {"error": "ConnectionError: down"},
                             "trading_log": {"facts": [], "data": {"notes": []}},
                             "account": {"error": "ConnectionError: down"}})
    text = R.job_report(ctx, "open", 0)
    assert "I couldn't read today's orders (ConnectionError: down)." in text
    assert "I couldn't read the account (ConnectionError: down)." in text


def test_study_report(ctx, monkeypatch):
    fake_tools(monkeypatch, {
        "study_report": {"facts": ["Last study term: Nov 17.", "Verdict: not ready yet."], "data": {}},
        "lessons": {"facts": ["Strength: I do best after a sharp selloff."], "data": {}}})
    assert R.job_report(ctx, "study", 0) == (
        "School's out for today.\nLast study term: Nov 17.\nVerdict: not ready yet.\n"
        "Newest lesson: Strength: I do best after a sharp selloff.")


def test_failure_report_includes_the_log_tail(ctx):
    (ctx.data_dir / "collect.log").write_text("==== Tue 11/17/2026  9:25:00.00 ====\nTraceback: boom\nExit code: 1\n")
    text = R.job_report(ctx, "collect", 1)
    assert text.startswith("Problem: the 0DTE data collector failed with exit code 1.")
    assert "Traceback: boom" in text and "data/collect.log" in text


def test_collect_success_sends_nothing(ctx):
    assert R.job_report(ctx, "collect", 0) is None


def test_unknown_job(ctx):
    with pytest.raises(ValueError):
        R.job_report(ctx, "dance", 0)


def test_missed_run_message(tmp_path):
    today = date(2026, 11, 17)
    assert "hasn't run today" in R.missed_run_message("open", tmp_path, today)
    (tmp_path / "open.log").write_text("==== Mon 11/16/2026  9:35:00.00 ====\nExit code: 0\n")
    assert "the 9:35 AM trading run hasn't run today" in R.missed_run_message("open", tmp_path, today)
    (tmp_path / "open.log").write_text("==== Tue 11/17/2026  9:35:00.00 ====\n")
    assert R.missed_run_message("open", tmp_path, today) is None


class FakePost:
    def __init__(self, status=200):
        self.calls, self.status = [], status

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        return httpx.Response(self.status, json={}, request=httpx.Request("POST", url))


def test_send_posts_each_chunk_without_pinging_anyone():
    post = FakePost()
    assert N.send("tok", 222, "a" * 2500, post=post) == 2
    url, headers, body = post.calls[0]
    assert url == "https://discord.com/api/v10/channels/222/messages"
    assert headers["Authorization"] == "Bot tok" and headers["User-Agent"].startswith("DiscordBot (")
    assert body["allowed_mentions"] == {"parse": []} and len(body["content"]) == 2000


def discord_env(monkeypatch, ready=True):
    for k in ("DISCORD_BOT_TOKEN", "DISCORD_OWNER_ID", "DISCORD_CHANNEL_ID"):
        monkeypatch.delenv(k, raising=False)
    if ready:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setenv("DISCORD_OWNER_ID", "1")
        monkeypatch.setenv("DISCORD_CHANNEL_ID", "222")
    monkeypatch.setattr("agent.discord_bot.config.load_dotenv", None)


def test_notify_skips_without_settings(monkeypatch, capsys, ctx):
    discord_env(monkeypatch, ready=False)
    post = FakePost()
    assert N.main(["open", "0"], ctx=ctx, post=post) == 0
    assert "Discord not set up, skipping report." in capsys.readouterr().out
    assert post.calls == []


def test_notify_sends_a_failure_and_survives_a_discord_error(monkeypatch, capsys, ctx):
    discord_env(monkeypatch)
    ok = FakePost()
    assert N.main(["collect", "1"], ctx=ctx, post=ok) == 0
    assert ok.calls and "failed with exit code 1" in ok.calls[0][2]["content"]
    assert N.main(["collect", "1"], ctx=ctx, post=FakePost(status=500)) == 0
    assert "Couldn't send the Discord report" in capsys.readouterr().out


def test_notify_collect_success_sends_nothing(monkeypatch, ctx):
    discord_env(monkeypatch)
    post = FakePost()
    assert N.main(["collect", "0"], ctx=ctx, post=post) == 0 and post.calls == []


@pytest.mark.parametrize("bat,job", [("run_open.bat", "open"), ("run_study.bat", "study"),
                                     ("run_collect.bat", "collect")])
def test_bat_files_save_the_code_then_notify(bat, job):
    text = (ROOT / bat).read_text()
    assert 'set "CODE=%errorlevel%"' in text
    assert f"echo Exit code: %CODE% >> data\\{R.LOG_FILES[job]}" in text
    assert f"%PY% -m agent.discord_bot.notify {job} %CODE% >> data\\{R.LOG_FILES[job]} 2>&1" in text
    assert text.index('set "CODE=%errorlevel%"') < text.index("agent.discord_bot.notify")


def test_notify_bad_arguments_exit_zero(capsys):
    assert N.main(["bogus", "x"]) == 0
    assert "bad arguments" in capsys.readouterr().out
    assert N.main([]) == 0


# ---------------------------------------------------------------- the box breakout jobs
def box_ctx(tmp_path, state=None, results=None):
    from agent.stan.paper import PaperBook
    journal = tmp_path / "stan.db"
    book = PaperBook(journal)
    if state is not None:
        book.set("box_today", state)
    if results is not None:
        book.set("box_results", results)
    book.close()
    return T.ToolContext(journal_path=str(journal), data_dir=tmp_path, now=lambda: NOW)


SENT = {"day": "2026-11-17", "status": "sent", "reason": "the 09:40 bar closed at $102.00, above the high of the box",
        "box": [101.0, 100.0],
        "order": {"id": "stan-box-20261117", "qty": 500, "side": "buy", "stop": 100.0, "target": 108.0, "risk": 2.0}}


def test_box_report_when_the_order_went_out(tmp_path):
    assert R.job_report(box_ctx(tmp_path, SENT), "box", 0) == (
        "Box breakout for Nov 17:\n"
        "Box (9:00-9:30): high $101.00, low $100.00.\n"
        "Sent a bracket order to buy 500 QQQ at market: stop $100.00, target $108.00 (3R), "
        "risking about $1,000.00.\n"
        "Paper money only.")


def test_box_report_with_no_trade(tmp_path):
    state = {"day": "2026-11-17", "status": "no_signal", "box": [101.0, 100.0],
             "reason": "nothing closed outside the box ($101.00 high, $100.00 low) by 10:30"}
    text = R.job_report(box_ctx(tmp_path, state), "box", 0)
    assert "No trade today: nothing closed outside the box ($101.00 high, $100.00 low) by 10:30." in text
    assert R.job_report(box_ctx(tmp_path / "x"), "box", 0).startswith("Box breakout for Nov 17:\nNo record")


def test_flatten_report(tmp_path):
    done = {**SENT, "status": "closed", "result": {"day": "2026-11-17", "side": "buy", "qty": 500, "entry": 102.1,
                                                   "exit": 104.1, "r": 1.0, "pnl": 1000.0, "reason": "time"}}
    assert R.job_report(box_ctx(tmp_path, done), "flatten", 0) == (
        "Box trade for Nov 17: long 500 QQQ from $102.10, closed at 3:55 at $104.10: +1.00R ($1,000.00).\n"
        "Paper money only.")
    failed = {**SENT, "status": "flatten_failed", "reason": "the closing order hasn't filled yet"}
    assert "couldn't close the box position" in R.job_report(box_ctx(tmp_path / "f", failed), "flatten", 0)
    assert R.job_report(box_ctx(tmp_path / "n", {"day": "2026-11-17", "status": "no_box", "reason": "x"}),
                        "flatten", 0) is None


def test_box_job_failure_uses_its_log(tmp_path):
    ctx = box_ctx(tmp_path)
    (tmp_path / "box.log").write_text("==== Tue 11/17/2026  9:31:00.00 ====\nTraceback: boom\nExit code: 1\n")
    text = R.job_report(ctx, "box", 1)
    assert text.startswith("Problem: the 9:31 AM box breakout watch failed with exit code 1.") and "boom" in text
    assert R.LOG_FILES["flatten"] == "flatten.log"
