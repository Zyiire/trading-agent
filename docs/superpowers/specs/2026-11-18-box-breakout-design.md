# Box breakout: Stan's main strategy (design)

Date: 2026-11-18
Source: the "Box Breakout" prompt Zy supplied (NQ opening-range strategy from a video).
Decisions by Zy: trade it on **QQQ** in the Alpaca paper account; it **replaces the tiers** as
what the paper account trades (tiers stay selectable); trade it **regardless** of the backtest
(the backtest is Stan's report card); **1R = 1% of the account**; live design **option A**
(two scheduled jobs).

## Rules (the video's, applied to QQQ)

All times New York (`agent.stan.zerodte.bars.ET`). Bars are built from 1-minute bars.

1. **Box:** highest high and lowest low of the six 5-minute bars that open 9:00 to 9:25. Skip the
   day if any of the six is missing.
2. **Entry:** the first 5-minute bar opening 9:30 to 10:25 (so closing by 10:30) that closes above
   the box high (long) or below the box low (short). Enter at that bar's close (live: a market
   order right after it closes).
3. **Stop:** the opposite side of the box. 1R = distance from entry to stop.
4. **Target:** 3R.
5. **Time stop:** flat at 3:55 PM.
6. **No trade** if nothing closes outside the box by 10:30. **One trade a day.**
7. **Trend filter:** 15-minute bars from all available minutes (pre-market included); long only
   when EMA(50) > EMA(200), short only when below. Use only 15-minute bars that have finished
   by the breakout bar's close (no look-ahead).
8. Gaps decided from Stan's 2021-2026 backtest: **no** 1R cap (every cap tested cut the total),
   **trade** news days (skipping them lowered the average), a bar touching both stop and target
   counts as the **stop**.

## Backtest results that justify the rules (2026-11-18 spike, QQQ 2021-2026, 1 NQ point of cost)

771 trades, 37.0% wins, +0.177R a trade, profit factor 1.28, max drawdown -23.1R, +136.5R;
2021-22 +0.12R, 2023-now +0.21R; 2 points of cost +0.157R; Monte Carlo median drawdown 22.7R,
1-in-20 35.8R. At 1% risk, $1 became $3.53 (worst drop -21%) vs $2.09 holding QQQ (worst
drop -35.7%).

10-year rerun (QQQ SIP 1-minute bars 2016-2026, same rules), against the video's claims:

| | Video (NQ) | Stan (QQQ) |
|---|---|---|
| No trend filter | 2,727 trades, +0.08R, max DD -41R | 2,669 trades, +0.089R, max DD -38.2R |
| With trend filter | 1,371 trades, 37.3% wins, +0.17R, PF 1.23-1.26, max DD -26R, +232R | 1,327 trades, 36.2% wins, +0.139R, PF 1.21, max DD -32.9R, +184R |
| Positive years | 9 of 11 | 9 of 11 (2017 and 2020 negative) |
| Split | 2016-22 +0.15R, 2023-now +0.21R | 2016-22 +0.093R, 2023-now +0.211R |
| Monte Carlo | median DD 26.7R, 1-in-20 43.8R | median DD 31.4R, 1-in-20 48.4R |

2 points of cost: +0.095R (2023-now +0.192R). At 1% risk $1 became $5.27 (worst drop -29.3%)
vs $5.97 holding QQQ (worst drop -35.7%). The 1R caps and the news-day skip lowered the
average over 10 years too, so the gap decisions stand. The live and exam code fetch SIP bars.

## Components

| Unit | Purpose |
|---|---|
| `agent/stan/box/rules.py` | Pure functions: build 5- and 15-minute bars from 1-minute rows, the box, the breakout signal with the trend filter, sizing, and the exit simulation on 1-minute bars. Used by both the exam and live trading, so they can't drift apart. |
| `agent/stan/box/exam.py` | `box_exam(minute_rows) -> dict`: the report card (headline stats, by year, 2021-22 vs 2023-now split, 2x costs, target and flat-time grid, Monte Carlo of 1,000 reshuffles, money view at 1% risk vs holding QQQ, the video's claims side by side). Saved as journal meta `box_exam` during Stan's study term, with plain-English thoughts. |
| `agent/stan/box/live.py` | `python -m agent.stan.box.live watch`: 9:31 to 10:31, once a minute, builds today's box and signal from fresh Alpaca 1-minute bars and sends **one bracket order** (market entry, stop-loss, 3R take-profit, day order, whole shares). `python -m agent.stan.box.live flatten`: at 3:55 cancels Stan's open box orders and closes the box position, then records the trade's R. |
| Broker | Paper-only `submit_bracket_order(symbol, qty, side, stop, target, client_order_id)`, `cancel_orders(prefix)`, `close_position(symbol)`. |
| Paper account | Strategy `box` joins selloff/low/medium/high. With `box`, the 9:35 run sells any tier holdings once, places no tier or selloff orders, and still runs the kill switch. Default strategy becomes `box`. |
| Schedule | Tasks "Stan box run" (9:31 weekdays) and "Stan box flatten" (3:55 weekdays), via `run_box.bat` and `run_flatten.bat`, added to `schedule_trading.bat`. Both call `agent.discord_bot.notify` with new jobs `box` and `flatten`. |
| Discord | Read-only tool `box_breakout`: today's box, signal, order and result, plus the exam's facts. `strategy` names the box strategy. |
| Web app | "Box breakout" appears in the strategy picker; `/paper/use` accepts `box`. |

## Risk

- 1R = 1% of equity at entry; quantity = floor(1% of equity / 1R). If that position would cost more
  than 95% of buying power (a narrow box), **size down** to the largest whole-share quantity within
  95% of buying power, so that trade risks less than 1% (the reduced risk % is written to the paper
  log). Skip only if that is under 1 share. (Ruling 2026-11-19, replacing the earlier "skip".)
- The exam's "live sizing" money view shows the effect: each trade risks
  min(1%, (0.95 x 4 x equity / entry) x 1R / equity), i.e. buying power assumed to be 4x equity; it
  reports $1 growth, the worst drop and how many trades were sized down.
- One trade a day; skip after a weekly loss of 3R (Monday to Friday) or when trading is paused or
  halted; the existing 20% kill switch still applies.
- Paper only. The broker methods refuse a non-paper account.
- Every signal, order, fill and reason is written to the paper log, so Stan can explain it.

## Testing

Offline unit tests with synthetic minute bars for every rule (box, both directions, no trade by
10:30, trend filter blocking, stop-first on a both-touch bar, 3:55 exit, sizing, weekly limit),
a fake broker for live (bracket sent once, never twice, flatten closes and records R, dry run
sends nothing), and the existing read-only check for the Discord package.
