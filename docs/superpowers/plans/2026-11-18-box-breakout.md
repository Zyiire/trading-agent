# Box Breakout Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stan trades the opening-range box breakout on QQQ as the paper account's main strategy, sits a box exam during school, and can explain it in Discord.

**Architecture:** A new package `agent/stan/box/` with pure rules shared by the exam and live trading, a live module run by two scheduled jobs (9:31 entry watcher, 3:55 flatten), a `box` paper strategy, and a read-only Discord tool.

**Tech Stack:** Python 3.14, pandas/numpy, alpaca-py (paper, bracket orders), SQLite, pytest, Windows Task Scheduler.

**Spec:** `docs/superpowers/specs/2026-11-18-box-breakout-design.md`

## Global Constraints

- All times New York (`agent.stan.zerodte.bars.ET`). Box = 5-minute bars opening 9:00-9:25 (all six required); signal bars open 9:30-10:25; entry at the breakout bar's close; stop = opposite box side; target 3R; flat 3:55 PM; one trade a day; trend filter 15-minute EMA(50) vs EMA(200) using only 15-minute bars finished by the signal bar's close; a bar touching stop and target counts as the stop; no 1R cap; news days traded.
- 1R = 1% of equity at entry; whole shares; skip if qty < 1 or notional > 95% of buying power; skip after a weekly (Mon-Fri) loss of 3R; skip when paper trading is paused/halted; the existing 20% kill switch stays.
- Paper only: every new order method refuses a non-paper account. Client order ids start `stan-box-`.
- The Discord package `agent/discord_bot/` stays read-only (its AST test must keep passing).
- Minute data: Alpaca SIP 1-minute bars in `data/box_bars.db` (table `bars_1m`, same columns as `data/zerodte.db`), filled by `python -m agent.stan.box.data backfill` (2016-01-01 onward, resumable) and topped up on each live run.
- All tests offline (no Alpaca, Discord or Ollama). Run with `python -m pytest -q` from the repo root.
- Never print or commit `.env`.

## Review Focus

1. A day whose pre-open box is incomplete (holiday, half day, data gap): no trade, no crash.
2. The watcher restarting mid-window (crash, PC wake): it must never send a second entry the same day (client order id per day + book state).
3. A breakout whose 1R needs more shares than buying power allows: skip with a logged reason.
4. Flatten when the bracket already exited (stop or target hit): nothing to close, R still recorded from fills.
5. Strategy switched to `box` while tier holdings exist: they are sold once, not every day.

---

### Task 1: Rules, data and the box exam

**Files:** Create `agent/stan/box/__init__.py`, `rules.py`, `data.py`, `exam.py`; modify `agent/stan/train.py` (one new curriculum step that runs the exam and saves meta `box_exam`); tests `tests/test_box_rules.py`, `tests/test_box_exam.py`.

**Produces:** `rules.bars(minute_rows, minutes) -> DataFrame` (ET index, o/h/l/c); `rules.box(five_day) -> tuple[float, float] | None`; `rules.trend_up(fifteen, at) -> bool | None`; `rules.signal(five_day, box, trend) -> Signal | None` with `Signal(side, bar_open, entry, stop, target, risk)`; `rules.simulate_exit(minutes_after, signal, flat=time(15,55)) -> (exit_price, reason)`; `rules.size(equity, risk_per_share, buying_power, risk_pct=0.01) -> int`; `data.load(path) -> DataFrame`, `data.backfill(...)`; `exam.box_exam(minutes_df, cost_points=1.0) -> dict` with keys `headline`, `no_filter`, `cost_2x`, `by_year`, `split`, `grid`, `monte_carlo`, `money` (1% risk vs holding QQQ), `video` (the claimed numbers), `facts` (plain-English sentences, every number pre-formatted).

- [ ] Write failing tests with synthetic minute bars: box from six bars; missing bar -> None; long breakout; short breakout; no breakout by 10:30 -> None; trend filter blocks a counter-trend breakout and uses only finished 15-minute bars; stop-first on a both-touch bar; target exit; 3:55 time exit; sizing (1% rule, buying-power cap, < 1 share -> 0); exam on a small synthetic set produces every key and correct counts.
- [ ] Implement rules/data/exam (the spike script in the spec's history is the reference: results must match it — 1,327 trades and +0.139R on 2016-2026 SIP data).
- [ ] Hook into `train.py`: a "box" curriculum step that loads `data/box_bars.db` (skip with a thought if missing), runs `box_exam`, saves meta `box_exam`, and writes its `facts` as thoughts; the step must not break when the db is absent.
- [ ] Full suite green; commit.

### Task 2: Live trading, the `box` strategy, and the schedule

**Files:** Modify `agent/broker/base.py`, `agent/broker/alpaca.py` (paper-only `submit_bracket_order`, `cancel_orders(prefix)`, `close_position(symbol)`), `agent/stan/paper.py` (`box` in `set_strategy`, `--use`, and `run_evening`: sell tier holdings once when switching, no tier/selloff orders while `box`), `agent/discord_bot/reports.py` (jobs `box` and `flatten` in `JOB_NAMES`/`LOG_FILES` with fixed-wording reports built from the paper log and box state), `schedule_trading.bat`; create `agent/stan/box/live.py`, `run_box.bat`, `run_flatten.bat`; tests `tests/test_box_live.py` plus additions to existing tests.

- [ ] Failing tests with a fake broker: watch sends exactly one bracket (side, qty, stop, target, id `stan-box-YYYYMMDD`) when the signal appears; a second watch call the same day sends nothing; dry run sends nothing; skipped when halted/paused, weekly loss >= 3R, qty 0; flatten cancels `stan-box-` orders, closes the position, records R from fills; flatten with nothing open records the bracket's exit; `run_evening` with `box` sells tier holdings once; broker methods refuse non-paper.
- [ ] Implement. `live.py watch` loops once a minute from start until 10:31 (exits early once a trade is placed or the day is decided), fetching fresh SIP minute bars into `data/box_bars.db`. `live.py flatten` runs once. Both log every decision with `book.log` and keep state in paper meta `box_today` and `box_results`.
- [ ] `.bat` files follow `run_open.bat` (Python lookup, log header, save `CODE`, call `agent.discord_bot.notify <job> %CODE%`), CRLF. `schedule_trading.bat` adds "Stan box run" (09:31 weekdays) and "Stan box flatten" (15:55 weekdays).
- [ ] Full suite green; commit.

### Task 3: Stan knows the strategy (Discord + web + docs)

**Files:** Modify `agent/discord_bot/tools.py` (read-only tool `box_breakout`: today's box/signal/order/result from paper meta and log, plus `box_exam` facts; `STRATEGY_NAMES["box"] = "the box breakout on QQQ"`; `strategy` mentions the box exam headline), `agent/web.py` and `agent/webapp/index.html` (`/paper/use` accepts `box`; picker shows "Box breakout"), `README.md` (section on the box strategy); tests in `tests/test_discord_tools.py`, `tests/test_web.py`.

- [ ] Failing tests: `box_breakout` facts for exam present/absent and for today's trade states; the tool registry has 11 tools; read-only AST test still passes; `/paper/use` with `box` succeeds locally.
- [ ] Implement; full suite green; commit.
