"""Stan's training school. He studies up to 15 years of daily history, sits an
exam on every year he has not seen, and journals what he learns.

Run:  python -m agent.stan.train           (real market data)
      python -m agent.stan.train --demo    (made-up practice data, to try the app)

This never places a trade. It only reads price history.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from agent.stan.journal import Journal
from agent.stan.persona import pct, signed_pct
from agent.stan import daytrade
from agent.stan.strategies import stress_test
from agent.stan.study import StudyConfig, build_panel, find_lessons, luck_test, summarize, walk_forward

# The 8 stocks from Stan's first term. Several were picked knowing they became huge
# winners, so the stress tests check his strategy on the other stocks separately.
ORIGINAL_EIGHT = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "IWM")
# Term 2 universe: big, liquid names from every sector, including long-time laggards
# (INTC, PFE, BA, MMM, T, VZ, F, C), so the test is not only past winners.
DEFAULT_SYMBOLS = ORIGINAL_EIGHT + (
    "DIA", "AMZN", "GOOGL", "META", "JPM", "BAC", "WFC", "GS", "C", "XOM", "CVX",
    "JNJ", "PFE", "MRK", "UNH", "KO", "PEP", "PG", "WMT", "HD", "MCD", "NKE", "DIS",
    "INTC", "CSCO", "ORCL", "IBM", "T", "VZ", "BA", "CAT", "GE", "MMM", "F", "GM",
)
CURRICULUM = [
    ("collect", "Collect 15 years of price history"),
    ("features", "Turn prices into signals I can study"),
    ("regimes", "Learn the eras: bull, bear and sideways years"),
    ("exams", "Sit an exam on every year I have not seen"),
    ("review", "Review my wins and losses"),
    ("selloff", "Strategy exam: buy after a sharp selloff"),
    ("stress", "Stress-test the selloff strategy"),
    ("daytrade", "Day-trading exam: opening-range breakout"),
    ("report", "Write my report card"),
]


# ---------- data ----------
def demo_history(symbols: Sequence[str], years: int = 15, seed: int = 11) -> dict[str, pd.DataFrame]:
    """Made-up prices with bull and bear stretches. For trying the app only."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=years * 252)
    regime, market = 1, []
    for _ in days:
        if rng.random() < (0.004 if regime == 1 else 0.012):
            regime *= -1
        market.append((0.0007 if regime == 1 else -0.0011) + rng.normal(0, 0.009 if regime == 1 else 0.016))
    market = np.array(market)
    history = {}
    for symbol in symbols:
        beta = rng.uniform(0.8, 1.6)
        r = beta * market + rng.normal(0, 0.012, len(days))
        close = 100 * np.cumprod(1 + r)
        spread = np.abs(rng.normal(0, 0.008, len(days)))
        history[symbol] = pd.DataFrame({
            "open": close * (1 - r / 2), "high": close * (1 + spread), "low": close * (1 - spread),
            "close": close, "volume": rng.lognormal(15, 0.4, len(days)) * (1 + 20 * np.abs(r)),
        }, index=days)
    return history


def _from_yfinance(symbols: Sequence[str], years: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf  # optional dependency

    history = {}
    for symbol in symbols:
        df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        history[symbol] = df
    return history


def _from_store(symbols: Sequence[str], db_path: str | Path) -> dict[str, pd.DataFrame]:
    """Bars already downloaded from Alpaca by `agent.jobs.backfill_bars`."""
    if not Path(db_path).exists():
        return {}
    history = {}
    with sqlite3.connect(str(db_path)) as db:
        for symbol in symbols:
            df = pd.read_sql_query(
                "SELECT day, open, high, low, close, volume FROM daily_bars WHERE symbol = ? ORDER BY day",
                db, params=(symbol,), parse_dates=["day"], index_col="day",
            )
            if len(df):
                history[symbol] = df
    return history


def real_history(symbols: Sequence[str], years: int, db_path: str | Path) -> tuple[dict[str, pd.DataFrame], str]:
    try:
        history = _from_yfinance(symbols, years)
        if history:
            return history, "Yahoo Finance"
    except Exception:
        pass
    return _from_store(symbols, db_path), "Alpaca (local database)"


# ---------- the curriculum ----------
def run_training(
    journal: Journal,
    history: dict[str, pd.DataFrame],
    source: str,
    demo: bool = False,
    cfg: StudyConfig = StudyConfig(),
    pace: float = 0.0,
    intraday: dict[str, str] | None = None,
) -> dict:
    """Run every task, writing progress and thoughts to the journal. Returns the summary."""

    def pause() -> None:
        if pace > 0:
            time.sleep(pace)

    journal.reset()
    journal.plan_tasks(CURRICULUM)
    journal.set_meta("run_id", time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000000}")
    journal.set_meta("mode", "demo" if demo else "real")
    journal.set_meta("stage", "student")
    journal.set_meta("state", "training")
    journal.think("Day one of school. No trades until I pass my exams. Let's study.", "milestone")
    if demo:
        journal.think("Heads up: this is practice data, not real market history. "
                      "The numbers show how I work, not how real markets behave.", "thought")

    # 1. collect
    journal.start_task("collect", f"Loading history from {source}")
    if not history:
        msg = ("I could not load any price history. Run `python -m agent.jobs.backfill_bars` first, "
               "or install yfinance, then try again.")
        journal.finish_task("collect", msg, failed=True)
        journal.think(msg, "fail")
        journal.set_meta("state", "stopped")
        return {}
    spans = {s: (df.index.min(), df.index.max(), len(df)) for s, df in history.items()}
    for i, (symbol, (start, end, n)) in enumerate(spans.items(), start=1):
        journal.update_task("collect", i / len(spans), f"{symbol}: {n:,} trading days")
        pause()
    years_have = min((end - start).days / 365.25 for start, end, _ in spans.values())
    total_days = sum(n for _, _, n in spans.values())
    journal.finish_task("collect", f"{len(spans)} stocks, {total_days:,} trading days, {years_have:.1f} years")
    journal.think(f"I have {total_days:,} days of prices across {len(spans)} stocks from {source}, "
                  f"covering {years_have:.1f} years.", "progress")
    if years_have < 14.5 and not demo:
        journal.think(f"That is short of the 15 years I wanted. This source only gave me {years_have:.1f}. "
                      "I'll study what I have and say so on my report card.", "thought")

    # 2. features
    journal.start_task("features", "Computing trend, momentum, volatility and volume signals")
    panel = build_panel(history)
    journal.finish_task("features", f"{len(panel):,} study rows, 14 signals each")
    journal.think(f"Built {len(panel):,} study rows. Each one holds only what I could have known "
                  "at that day's close. No peeking at tomorrow.", "progress")
    pause()

    # 3. regimes
    journal.start_task("regimes", "Reading each year's character")
    bench_symbol = "SPY" if "SPY" in history else next(iter(history))
    bench = history[bench_symbol]["close"].sort_index()
    by_year = list(bench.groupby(bench.index.year))
    counts = {"bull": 0, "bear": 0, "sideways": 0}
    for i, (year, closes) in enumerate(by_year, start=1):
        if len(closes) < 60:
            continue
        ret = closes.iloc[-1] / closes.iloc[0] - 1
        drop = float((closes / closes.cummax() - 1).min())
        label = "bull" if ret > 0.10 else "bear" if ret < -0.10 else "sideways"
        counts[label] += 1
        journal.update_task("regimes", i / len(by_year), f"{year}: {label}")
        journal.think(f"{year}: {bench_symbol} went {signed_pct(ret)} with a worst drop of {pct(drop)}. "
                      f"I'm filing that as a {label} year.", "thought")
        pause()
    journal.finish_task("regimes", f"{counts['bull']} bull, {counts['bear']} bear, {counts['sideways']} sideways years")
    journal.think("Mostly up years means always guessing 'up' looks smart. "
                  "I have to beat that, not just beat a coin flip.", "thought")

    # 4. exams
    journal.start_task("exams", "Training on the past, testing on the next unseen year")

    def on_year(exam: dict, trades: list[dict], i: int, total: int) -> None:
        journal.record_exam(exam)
        journal.update_task("exams", i / total, f"Exam {i} of {total}: {exam['year']}")
        if exam["trades"] == 0:
            journal.think(f"{exam['year']} exam: I was never confident enough to trade, so I sat in cash. "
                          f"Holding did {signed_pct(exam['hold_return'])}.", "thought")
        else:
            verdict = "Passed." if exam["passed"] else "Failed."
            versus = "beat" if exam["beat_hold"] else "trailed"
            journal.think(
                f"{exam['year']} exam: {exam['trades']} trades, won {pct(exam['win_rate'])}, "
                f"profit factor {exam['profit_factor']}. I made {signed_pct(exam['stan_return'])} and "
                f"{versus} simply holding ({signed_pct(exam['hold_return'])}). {verdict}",
                "win" if exam["passed"] else "fail", exam,
            )
        if trades:
            worst = min(trades, key=lambda t: t["ret"])
            best = max(trades, key=lambda t: t["ret"])
            if worst["ret"] < 0:
                journal.think(f"Worst trade of {exam['year']}: {worst['symbol']} on {worst['entry']:%b %d}, "
                              f"{signed_pct(worst['ret'])} in {worst['days']} day(s). Logged for review.", "fail")
            if best["ret"] > 0:
                journal.think(f"Best trade of {exam['year']}: {best['symbol']} on {best['entry']:%b %d}, "
                              f"{signed_pct(best['ret'])} in {best['days']} day(s).", "win")
        pause()

    exams, trades, daily = walk_forward(panel, cfg, on_year)
    if not exams:
        msg = (f"I need at least {cfg.min_train_years} years of history before my first exam, "
               "and I don't have that yet.")
        journal.finish_task("exams", msg, failed=True)
        journal.think(msg, "fail")
        journal.set_meta("state", "stopped")
        return {}
    journal.update_task("exams", 1.0, "Checking whether random entries would have done as well")
    closes = {sym: df.sort_index()["close"].astype(float) for sym, df in history.items()}
    luck = luck_test(trades, closes, cfg.cost_per_side, reps=cfg.luck_reps)
    summary = summarize(exams, trades, daily, luck)
    journal.finish_task("exams", f"{summary['years_passed']} of {summary['years_tested']} exams passed")
    if luck["p_value"] is not None:
        beat = luck["p_value"] < 0.05
        journal.think(
            f"Luck test: my trades averaged {signed_pct(luck['actual_avg'], 2)} each. The same number of "
            f"random entries in the same stocks and years averaged {signed_pct(luck['random_avg'], 2)} "
            f"(p = {luck['p_value']}). " + ("My picks beat random timing." if beat else
            "I can't tell my picks apart from random timing yet."),
            "win" if beat else "fail", luck)

    # 5. review
    journal.start_task("review", "Sorting my trades by market condition")
    lessons = find_lessons(trades, cfg)
    for lesson in lessons:
        e = lesson["evidence"]
        text = (f"Across {e['trades']} trades {e['when']}, I won {pct(e['win_rate'])} with an average of "
                f"{signed_pct(e['avg_return'], 2)} per trade. Overall I win {pct(e['overall_win_rate'])} and "
                f"average {signed_pct(e['overall_avg'], 2)}.")
        if lesson["kind"] == "mistake":
            text += " Next step: trade smaller or stand aside in these conditions, then retest."
        else:
            text += " Next step: check this holds on new data before I lean on it."
        journal.add_lesson(lesson["kind"], lesson["title"], text, e)
        journal.think(f"Lesson: {lesson['title']}. {text}", "lesson")
        pause()
    if not lessons:
        journal.think("No market condition stands out clearly from my average, so I have no fair lesson "
                      "to draw yet.", "thought")
    journal.finish_task("review", f"{len(lessons)} lesson(s) written down")

    # 6. selloff strategy exam
    journal.start_task("selloff", f"Testing on {len(history)} stocks, every year")
    journal.think("Time for my first named strategy: the selloff bounce. When a stock's RSI closes under 30, "
                  "I buy at the next morning's open and sell when RSI recovers to 50 or after 10 days. "
                  "It needs no training, so every year is a fair test.", "milestone")
    strategy_checks: list[dict] = []

    def on_check(item: dict) -> None:
        strategy_checks.append(item)
        done = len(strategy_checks)
        task = "selloff" if done <= 2 else "stress"
        if done == 3:
            journal.finish_task("selloff", "Year-by-year results recorded")
            journal.start_task("stress", "Trying to break it")
        progress = done / 2 if task == "selloff" else (done - 2) / 8
        journal.update_task(task, min(1.0, progress), item["name"])
        if item["key"] == "years":
            for y in item["data"].get("by_year", []):
                if y["trades"]:
                    journal.think(f"Selloff bounce in {y['year']}: {y['trades']} trades, won {pct(y['win_rate'])}, "
                                  f"average {signed_pct(y['avg'], 2)} per trade.",
                                  "win" if (y["avg"] or 0) > 0 else "fail", y)
                    pause()
        verdict = "Passed" if item["passed"] else "Failed"
        journal.think(f"Stress test, {verdict.lower()}: {item['name']}. {item['detail']}.",
                      "win" if item["passed"] else "fail", {"check": item["key"]})
        pause()

    strategy = stress_test(history, list(ORIGINAL_EIGHT), on_check=on_check, luck_reps=cfg.luck_reps)
    journal.finish_task("stress", f"Passed {strategy['checks_passed']} of {strategy['checks_total']} stress tests")
    journal.set_meta("strategy", strategy)
    if strategy["verdict"] == "real_edge":
        journal.think("The selloff bounce survived every stress test I threw at it. That's my first real edge.",
                      "milestone")
    elif strategy["verdict"] == "promising":
        journal.think(f"The selloff bounce passed {strategy['checks_passed']} of {strategy['checks_total']} stress "
                      "tests. Promising, not proven. I'll study the ones it failed.", "lesson")
    else:
        journal.think(f"The selloff bounce passed only {strategy['checks_passed']} of {strategy['checks_total']} "
                      "stress tests. It isn't a real edge yet.", "fail")

    # 7. day-trading exam (Day Trader skill pack)
    journal.start_task("daytrade", "Loading 60 days of 5-minute bars")
    sessions = daytrade.exam_sessions(intraday or {})
    if not sessions:
        journal.finish_task("daytrade", "No 5-minute data, so I skipped this exam", failed=True)
        journal.think("I couldn't load any 5-minute bars, so the day-trading exam waits for next time.", "fail")
    else:
        n_sessions = max(len(v) for v in sessions.values())
        journal.think(
            f"New skill pack in my library: the Day Trader skill. Term 3 tests its opening-range breakout on "
            f"{len(sessions)} stocks over the last {n_sessions} sessions of 5-minute bars, exactly the way its "
            "Time Machine replays history. Only about 3 months of data exist for free, so this is one market mood, "
            "not 15 years.", "milestone")
        done = {"n": 0}

        def on_day_check(item: dict) -> None:
            done["n"] += 1
            journal.update_task("daytrade", done["n"] / 7, item["name"])
            journal.think(f"Day-trading exam, {'passed' if item['passed'] else 'failed'}: {item['name']}. "
                          f"{item['detail']}.", "win" if item["passed"] else "fail")
            pause()

        exam = daytrade.day_trade_exam(sessions, list(ORIGINAL_EIGHT), on_check=on_day_check,
                                       luck_reps=max(100, cfg.luck_reps // 3))
        journal.set_meta("daytrade_exam", exam)
        journal.finish_task("daytrade", f"Passed {exam['checks_passed']} of {exam['checks_total']} checks")
        if exam["verdict"] == "real_edge":
            journal.think("The opening-range breakout passed every check on these 60 days. One market mood is thin "
                          "evidence, so nightly practice has to confirm it.", "milestone")
        elif exam["verdict"] == "promising":
            journal.think(f"The opening-range breakout passed {exam['checks_passed']} of {exam['checks_total']} "
                          "checks. Promising, not proven.", "lesson")
        else:
            journal.think(f"The opening-range breakout passed only {exam['checks_passed']} of "
                          f"{exam['checks_total']} checks. On my stocks lately it isn't an edge, which is exactly "
                          "what the Time Machine is there to show before a dollar is risked.", "lesson")

    # 8. report
    journal.start_task("report", "Adding it all up")
    journal.set_meta("summary", summary)
    edge = summary["accuracy"] - summary["baseline_up"]
    journal.think(
        f"Report card: on {summary['years_tested']} unseen years I called direction right "
        f"{pct(summary['accuracy'])} of the time. Always guessing 'up' scores {pct(summary['baseline_up'])}, "
        f"so my edge on direction is {signed_pct(edge)}.", "progress")
    if summary["trades"]:
        journal.think(
            f"Trading: {summary['trades']} trades, won {pct(summary['win_rate'])}, profit factor "
            f"{summary['profit_factor']}. $1 with me became ${summary['stan_growth']:.2f}. "
            f"$1 simply held became ${summary['hold_growth']:.2f}.", "progress")
        extra = []
        if summary.get("spy_growth"):
            extra.append(f"$1 in SPY alone became ${summary['spy_growth']:.2f}")
        if summary.get("max_dd") is not None:
            extra.append(f"my worst drop was {pct(summary['max_dd'])} vs {pct(summary['hold_max_dd'])} holding")
        if summary.get("per_day_invested") is not None:
            extra.append(f"I was invested {pct(summary['exposure'])} of the time, earning "
                         f"{signed_pct(summary['per_day_invested'], 3)} per day invested vs "
                         f"{signed_pct(summary['per_day_hold'], 3)} per day holding")
        if extra:
            journal.think("Fair comparison: " + "; ".join(extra) + ".", "progress")
    if summary["ready_for_paper"] or strategy["verdict"] == "real_edge":
        journal.set_meta("stage", "paper_candidate")
    if summary["ready_for_paper"]:
        verdict = ("I cleared my bar: I win at least half my trades, make more than I lose, and beat "
                   "holding after costs. I'm ready to try paper trading. Still no real money.")
        journal.think(verdict, "milestone")
    else:
        reasons = []
        if summary["win_rate"] is None or summary["win_rate"] < 0.5:
            reasons.append("I win fewer than half my trades")
        if (summary["profit_factor"] or 0) <= 1:
            reasons.append("I lose more than I make")
        if summary["stan_growth"] <= summary["hold_growth"]:
            reasons.append("simply holding beat me")
        verdict = ("My all-round model isn't ready to trade: " + "; ".join(reasons) + ". "
                   "That's the honest result, and I keep working on it.")
        journal.think(verdict, "fail")
    if not summary["ready_for_paper"] and strategy["verdict"] == "real_edge":
        journal.think("My overall model still loses to simply holding, but the selloff bounce passed every "
                      "stress test. That earns it a paper-trading trial on its own. Still no real money.",
                      "milestone")
    ready_any = summary["ready_for_paper"] or strategy["verdict"] == "real_edge"
    journal.finish_task("report", "Ready for paper trading" if ready_any else "Not ready yet")
    journal.set_meta("state", "finished")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stan's training school.")
    parser.add_argument("--demo", action="store_true", help="use made-up practice data")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--journal", default="data/stan.db")
    parser.add_argument("--db", default="data/agent.db", help="Alpaca bars database (fallback source)")
    parser.add_argument("--pace", type=float, default=None,
                        help="seconds to pause between steps so you can watch along in the app")
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.demo:
        history, source = demo_history(symbols, args.years), "practice data (demo)"
    else:
        history, source = real_history(symbols, args.years, args.db)
    pace = args.pace if args.pace is not None else (0.6 if args.demo else 0.2)

    intraday = daytrade.load_charts(symbols, demo=args.demo)
    journal = Journal(args.journal)
    summary = run_training(journal, history, source, demo=args.demo, pace=pace, intraday=intraday)
    for t in journal.thoughts(limit=1000):
        print(f"[{t['kind']:>9}] {t['text']}")
    journal.close()
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
