# Stan: AI Trading Agent

Stan is a trader in training. Before he trades anything, he goes to school:
he studies up to 15 years of price history, sits an exam on every year he has
not seen, and journals what his wins and losses teach him. A phone app shows
his thinking and task progress live.

**Nothing here can trade.** There is no order code yet, the Alpaca connection is
read-only, and live mode is locked in `agent/config.py`.

## Automatic trading, Monday to Friday

Double-click `schedule_trading.bat` once. It creates a Windows scheduled task that runs Stan's
paper trade at 9:35 AM every weekday, with no end date, using the same Python as your terminal.

- The PC must be on and awake at 9:35 AM.
- To pause Stan, flip **Trading** off in the web app. The scheduled run then places no orders.
- To remove the schedule: `schtasks /Delete /TN "Stan open run" /F`
- Each run is logged in `data/open.log`, including which Python ran and its exit code.
- It also schedules **Stan nightly study** at 6 PM every weekday (`run_study.bat`): a new school
  term on real history, logged in `data/study.log`. It places no orders, but it can change which
  strategy Stan ranks best, and that is what the next 9:35 AM run trades. Check the Classroom page.
  To stop it: `schtasks /Delete /TN "Stan nightly study" /F`

## Talk to Stan on Discord

Write to Stan in his Discord channel in plain English: "how did today go?", "what did you learn
last night?", "what are the QQQ levels?", "did everything run?". He answers from his real data
with a free AI model that runs on this PC (Ollama), and he posts a short report after the
9:35 AM trade, after the 6 PM study term, and if a job fails or didn't run.

He can only talk and report. Trading, pausing, changing tier and studying stay in the web app
and the automatic schedule.

Already installed Stan before this? Run `pip install -r requirements.txt` again (it adds discord.py).

One-time setup:

1. Ollama: install it from https://ollama.com and run `ollama pull qwen3:14b` (about 9.3 GB).
2. Discord Developer Portal (https://discord.com/developers/applications): **New Application**
   "Stan" > **Bot** > **Reset Token** > copy it. On the same page, turn on **Message Content Intent**.
3. **OAuth2** > **URL Generator**: tick **bot**, then **View Channels**, **Send Messages** and **Read
   Message History**. Open the link and add Stan to your private server.
4. In Discord: **User Settings** > **Advanced** > **Developer Mode** on. Right-click your name >
   **Copy User ID**; right-click Stan's channel > **Copy Channel ID**.
5. Put the three values in `.env`: `DISCORD_BOT_TOKEN`, `DISCORD_OWNER_ID`, `DISCORD_CHANNEL_ID`.
6. Double-click `schedule_trading.bat` once more. It adds the **Stan Discord** task, which starts
   the bot now and whenever you log in.

- Stan answers only you, only in that channel.
- Don't start `run_discord.bat` by hand while the **Stan Discord** task is running, or you'll get two bots.
- Ask a question from a terminal without Discord: `python -m agent.discord_bot.brain "how did today go?"`
- Faster but less careful brain: set `STAN_MODEL=qwen3:8b` in `.env`.
- The bot's log is `data/discord.log`. To stop it for good: `schtasks /Delete /TN "Stan Discord" /F`

## Stan on your PC (the web app)

Double-click `stan.bat` (or run `python -m agent.web`). Your browser opens http://127.0.0.1:8000.

| Page | What it does |
|---|---|
| **Overview** | Classroom progress, the live paper account with Today / 1W / 1M / All, what Stan is thinking (a preview of his next trade, refreshed on its own), his latest lessons, and the current plan |
| **Classroom** | The term's curriculum, Stan's notes as he works, strategy rankings, exam verdicts. **Start a new term** runs his school in the background (press twice to confirm) |
| **Trade journal** | Lessons, trading notes, daily results calendar, finished trades |
| **Paper account** | Live chart of the account or any holding, positions, and **Stan's trade**: pick Steady, Balanced, Aggressive or Selloff bounce, **Preview**, then **Send these orders** |
| **Settings** | Trading on/off (off also stops the scheduled morning run), plan, auto-stop level, link to Alpaca |

- Nothing is sent until you press Send, and Send only unlocks after a preview that contains orders.
- `stan.bat --demo` opens the same pages on a made-up account and prices.
- `stan.bat --phone` also lets phones on your Wi-Fi open the page (the address is printed). Other
  devices can look but not change anything: the trade buttons and switches are hidden there.
- Paper account only. Stan's portrait is `agent/webapp/stan.png`; replace that file to change it.

## Watch Stan study (start here)

Do the Python setup in the next section first (the Alpaca keys are only needed for the Alpaca commands, not for Stan's school).

Terminal 1, Stan's server (the app reads from it):

```bash
python -m agent.api
```

Terminal 2, send Stan to school:

```bash
python -m agent.stan.train --demo   # made-up practice data, to try the app
python -m agent.stan.train          # real market history
```

Terminal 3, the iPhone app (needs Node.js and the Expo Go app on your phone):

```bash
cd stan-app
npm install
cp .env.example .env    # put your computer's Wi-Fi address in it
npx expo start          # scan the QR code with your iPhone camera
```

Your phone and computer must be on the same Wi-Fi. The app has four tabs:
**Now** (current task and progress), **Mind** (Stan's live thought feed),
**Report card** (yearly exams, verdict, lessons) and **Setup**.

### How Stan learns

1. Collects price history (Yahoo Finance for the full 15 years; falls back to
   the Alpaca bars in the local database, which start around 2016).
2. Builds 14 signals per day using only what was known at that day's close.
3. Trains on the past, then is tested on the next year he has never seen,
   repeated year by year. Costs are charged on every trade.
4. Sorts his trades by market condition and writes down where he loses and wins.
5. Writes a report card. He only earns the "Paper Trader (candidate)" rank if he
   wins at least half his trades, makes more than he loses, and beats simply
   holding after costs.

### Term 2: stress tests

Stan now studies 43 big stocks from every sector (including long-time laggards
like INTC, PFE, BA and T), not just 8 famous winners. His report card adds:

- **Fair comparison:** SPY alone, time invested, return per day invested, and worst drop.
- **Luck test:** would random entries in the same stocks and years have done as well?
- **Strategy exam:** his "selloff bounce" (buy at the next open after RSI closes
  under 30, sell when RSI reaches 50 or after 10 days), which must survive 10
  stress tests: basic profitability at 0.10% costs, 70% of years profitable, the
  luck test, 0.25% costs, entry timing, 12 nearby settings, stocks he never
  studied, both market trends, return per day invested, and worst drop.

He earns the "Paper Trader (candidate)" rank if his all-round model beats holding,
or if the selloff bounce passes every stress test. A real run takes a few minutes.

### Shadow trading (practice on live days, no orders)

Once a day after the market closes, Stan:

1. **Follows the selloff bounce live** on days that happen after shadowing starts,
   and compares the results with what the backtest promised. Closed trades are
   recorded once and never rewritten.
2. **Makes beat-or-lag calls.** To break the "always guess up" habit, he no longer
   calls whether stocks go up. Every day he names 5 stocks he expects to **beat**
   SPY tomorrow and 5 he expects to **lag** it, then grades yesterday's calls.
   About half of all stocks beat the market on any day, so always saying "beat"
   earns nothing, and he always makes as many lag calls as beat calls.

Run it by hand:

```bash
python -m agent.stan.shadow          # one daily run (after 4:30pm ET)
python -m agent.stan.shadow --demo   # replay 40 days of practice data
```

To run it automatically on Windows: open **Task Scheduler**, choose **Create Basic
Task**, set it to **Daily** at **4:45 PM**, pick **Start a program**, and browse to
`run_shadow.bat` in this folder. Then open the task's properties and, under
**Triggers**, edit it to repeat on weekdays only. The PC has to be on (asleep is
fine if you tick **Wake the computer to run this task** under **Conditions**).
Output goes to `data/shadow.log`, and results show in the app's **Shadow** tab.

Live results need roughly 100 closed trades (about 3 to 4 months) before they
say much. Rerunning Stan's school never erases his shadow history.

### Term 3: the Day Trader skill pack

Stan's library now includes [The Day Trader skill](https://github.com/Jakeschincariol/day-trader-skill)
by Jake Schincariol (MIT), copied unchanged into `agent/skills/day_trader/` (version
and license in its `NOTICE.md`). The app's **Now** tab lists all 11 skills and how
Stan uses each one:

| Skill | Status | How Stan uses it |
| --- | --- | --- |
| Chart Scanner, Entry Timer, Exit Planner | in use | The opening-range breakout: buy on the first 5-minute close above the first 15 minutes' high, stop at the range midpoint, 1R target, out by 3:55 PM |
| Time Machine | in use | 10,000 replays before every practice trade (earlier sessions only), plus its history replay as his day-trading exam |
| Budget Manager, Risk Manager | in use | Size and check every practice trade on the $50 plan ($5 max loss, 3 trades and $10 risk a day) |
| Order Executor | locked | Stan can't place orders until he passes the go-live checks |
| Price Alerts, News Monitor, Earnings Analyst, Pelosi Tracker | learned | In his library, not wired into decisions yet |

- **Day-trading exam** (part of `python -m agent.stan.train`): the breakout on all 43
  stocks over the last 60 days of free 5-minute bars, with 7 checks (costs, a luck test,
  a 2R target, how many stocks it works on, week by week, new stocks). Free 5-minute
  data only goes back 60 days, so this is one market mood.
- **Day-trade practice** (part of `python -m agent.stan.shadow`, every evening): each
  breakout is walked through the pack (plan, Time Machine, budget, risk) and then graded
  against what actually happened. The **Shadow** tab tracks whether the Time Machine's
  odds come true. With $50, the budget manager refuses most stocks: one whole share
  costs more than the plan allows.

### Paper trading (Alpaca paper account)

Stan trades the **selloff bounce** on your Alpaca **paper** account with whatever paper
money it holds. This was your call: he passed 9 of 10 stress tests, not all 10, and the
app says so.

- After the close he places **market-on-open** orders (filled in the next opening auction,
  like the backtest): buy after RSI closes under 30; sell after RSI closes at 50+ or 10
  trading days pass.
- **Sizing:** 10% of the account per trade, up to 10 positions, whole shares.
- **Kill switch:** at 20% below the starting balance he sells everything at the next open
  and stops. Only you can restart him (`--resume`).
- **Safety:** every evening he checks his records against Alpaca's positions and stops on
  any mismatch. Each order has a fixed ID, so it can't be sent twice. He leaves alone any
  stock the account already holds outside his trades. Live trading stays locked in code.

```bash
python -m agent.stan.paper --dry-run   # see what he would do, send nothing (try this first)
python -m agent.stan.paper             # the real evening run (after 4:30pm ET)
python -m agent.stan.paper --status    # account, positions, halted or running
python -m agent.stan.paper --resume    # restart after the kill switch or a mismatch
```

`run_shadow.bat` now runs shadow trading and then paper trading, so the same 4:45 PM
scheduled task covers both. Results show at the top of the app's **Trades** tab.

Stan's thoughts are written from the real numbers his training produces. They
are templates filled with results, not a chatbot. A local language model for
richer explanations comes later in the plan.

Limits to know: this round studies daily prices (free 15-year intraday data does
not exist), so it teaches next-day trades. Intraday study comes next, using
Alpaca minute bars.

## Alpaca setup (about 5 minutes)

Needs Python 3.10 or newer.

```bash
cd trading-agent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Open `.env` and paste in your Alpaca **paper** API key and secret
(Alpaca dashboard, Paper account, API Keys). Keep `.env` private.

## Run it

```bash
python -m pytest -q                          # offline tests, no keys needed
python -m agent.jobs.check_connection        # 1. confirm the agent sees your account
python -m agent.jobs.backfill_bars           # 2. download daily price history
python -m agent.jobs.snapshot_chains --once  # 3. save one options snapshot
python -m agent.jobs.snapshot_chains         # 3b. keep saving every 15 min in market hours
```

`check_connection` prints your paper equity, options level and account type, and
lists anything to fix (for example: options not enabled, or the paper balance not
set to $50).

## What is here

| Path | Purpose |
| --- | --- |
| `agent/config.py` | Loads settings from `.env`; refuses live mode |
| `agent/broker/base.py` | Broker-neutral interface, so Robinhood can be added later |
| `agent/broker/alpaca.py` | Alpaca adapter (read-only) |
| `agent/store.py` | SQLite tables: daily bars, option snapshots, account snapshots |
| `agent/jobs/` | Alpaca connection check, price download, options snapshots |
| `agent/stan/` | Stan: persona, journal, study method, training school, strategies, shadow trading |
| `agent/skills/day_trader/` | The Day Trader skill pack, vendored unchanged (MIT) |
| `agent/api.py` | Read-only API for the phone app |
| `stan-app/` | Expo (React Native) iPhone app |
| `tests/` | Offline tests using fake Alpaca clients |

## The QQQ trend bot (from the reel)

`agent/stan/qqq_trend.py` is Stan's rebuild of the "QQQ BOTS" shown in Nick Fischer's
Instagram reel. The reel shows the dashboard switches (trend filter, chop filter, stop loss,
profit lock, trailing stop, daily loss limit, risk Low/Medium/High) and about one week of
results, but never the entry rule or a backtest. So this is a reconstruction from those
switches, not his code.

The rule, on 5-minute bars of the stock itself (options are not modelled):

- Long above VWAP with EMA9 over EMA21 and rising; short on the mirror (puts, in options terms).
- Chop filter: only when the last hour moved in a fairly straight line (efficiency ratio 0.30+).
- Stop 1.5 ATR. At +1R the stop moves to break-even, then trails the best price by 2 ATR.
- Out on a close back across VWAP, or at 3:55 PM. At most 3 trades a day, none after -2R.

Training runs it as its own exam with 9 checks: profit after costs, beating random entries,
higher costs, the short side alone, all three risk levels, the chop filter, weeks ahead,
breadth beyond QQQ, and drawdown. It shows on the Report tab. It does not trade until it
passes; if it does, the next step is shadow practice, then the paper account.

## Price action at key levels

`agent/stan/price_action/` turns Zy's discretionary playbook into fixed rules on raw OHLCV
(no RSI, no MACD). One module per part:

| File | Job |
|---|---|
| `config.py` | `PriceActionConfig`: every threshold (pivot N, wick ratio, zone buffer, R:R, risk, session times) |
| `series.py` | 5-minute bars as arrays; 15-minute bars built only from closed 5-minute bars |
| `structure.py` | Swing pivots confirmed only after N later bars close; HH/HL = up, LH/LL = down, else chop |
| `levels.py` | Previous day high/low, premarket high/low (when available), recent swing levels, zone buffer |
| `patterns.py` | Pin bar (2:1 wick, closes in the top/bottom 30%), engulfing on above-average volume, failed breakout |
| `risk.py` | Stop 2 cents past the wick, size = dollar risk / stop distance, 2R or next level, break-even at 1R |
| `engine.py` | The decision at each closed bar, and the trades it makes |
| `exam.py` | 9 stress tests, shown on the Report tab |

No lookahead: `tests/test_price_action.py` re-runs the engine on data cut off at bar i and checks
that the trend, pivots, levels and signal at bar i are identical to the full run. A planted
lookahead bug makes that test fail. A 15-minute bar that is still forming is never used.

## Stan's portfolio (staged)

`agent/stan/portfolio.py` builds a portfolio from four edges with long research records,
adding one stage at a time:

1. **Trend core:** hold SPY only while it is above its 10-month average at a month end, otherwise cash.
2. **Volatility sizing:** hold less SPY when the last month was wild (15% volatility target).
3. **Selloff bounce:** Stan's own rule as a 30% sleeve, 10% per position, at most 10 open.
4. **Momentum:** a 30% sleeve holding the top 5 stocks by 12-1 month return (if positive), monthly.

A stage stays only if its Sharpe ratio is no worse over the whole history and in each half,
and it either raises the Sharpe ratio or makes the worst drop 5+ points shallower. Allocations
are fixed in advance, not tuned. Costs are 0.10% per side, and cash earns nothing.

No lookahead: decisions use month-end closes, fill at the next close and earn from the day
after. `tests/test_portfolio.py` checks every month-end decision against data cut off on that
day, and catches a planted 5-day peek.

Caveat: the 43 stocks are today's well-known names (survivorship bias), so the momentum and
selloff sleeves look better here than they would have in real time.

The paper account keeps trading the selloff bounce until the portfolio passes on real data.

## Strategy arena (low, medium, high risk)

`agent/stan/arena.py` runs 16 strategies on the same daily history with the same costs and
scoring, then ranks them and picks a winner for each risk tier.

| Family | Strategies |
|---|---|
| Benchmarks | Hold SPY; 60/40 SPY/AGG; 3x QQQ always (raw leverage) |
| Trend and risk control | Trend core; volatility target; dual momentum (GEM); sector momentum |
| Option selling | Cash-secured SPY puts (like Cboe PUT); covered calls on SPY (like Cboe BXM) |
| Option buying | SPY calls; QQQ puts; SPY calls + QQQ puts; calls in uptrends / puts in downtrends |
| Stan's own | Selloff bounce; stock momentum (both flagged for survivorship bias) |
| Leverage | 3x QQQ with a 200-day trend switch (Gayed & Bilello) |

- **Tiers** come from the measured worst drop: low up to 20%, medium 20-40%, high deeper than 40%.
- **To win a tier**, a strategy must beat T-bills in both halves of the history and at double
  costs, and not be flagged for survivorship bias. Low and medium winners have the best Sharpe
  ratio; the high winner has the best yearly return with a Sharpe of 0.4+.
- **If nothing is low risk on its own**, Stan runs the best eligible strategy at a smaller size
  (rest in T-bills) so its worst drop stays within 20%.
- **Options** are priced with Black-Scholes from VIX (SPY) and VXN (QQQ) times 0.92, with 2% of
  the premium in costs. There is no free history of real option quotes, so these are estimates.
- **Extra data** (sector ETFs, EFA, AGG, VIX, VXN, T-bill rate) downloads from Yahoo during training.
- **Benchmarks** (Hold SPY, 3x QQQ always) are ranked but can never win a tier.
- **Re-rank only:** `python -m agent.stan.arena` re-runs the arena without the whole school.

### Choosing what the paper account trades

```
python -m agent.stan.paper --use low      # SPY sized to 15% volatility, at the arena's reduced size
python -m agent.stan.paper --use medium   # SPY sized to 15% volatility, full size
python -m agent.stan.paper --use high     # TQQQ while QQQ is above its 200-day average, else cash
python -m agent.stan.paper --use selloff  # back to the selloff bounce
```

The choice is saved, so the nightly `run_shadow.bat` keeps using it. Switching away from the
selloff bounce lets its open trades finish first. Orders are fractional market orders that fill
at the next open, sent only when a holding is more than 5% away from its target. The 20% kill
switch still sells everything and stops. While a tier is on, Stan manages the account's whole
SPY and TQQQ positions. Try `--dry-run` first to see the orders without sending them.

## 0DTE QQQ options bot (tz.fund style), in phases

Built from `claude_prompt_tzfund.md`: buy calls and puts on same-day QQQ options at key levels,
using put/call flow and gamma walls, with strict risk rules. Each phase stops for review.

**Phase 2 (done): the data pipeline, free sources only.** Everything lands in `data\zerodte.db`.

| Data | Source (free) | Notes |
|---|---|---|
| QQQ 1-minute bars, rolled into 3-minute bars with session VWAP | Alpaca full-market history since 2016 (the backfill starts in 2021); live from Alpaca IEX, Yahoo as backup | The free plan skips the latest 15 minutes of full-market data, so live bars come from IEX |
| Option chain: bid/ask, IV, greeks, day volume | Alpaca indicative feed, Yahoo as backup | Indicative quotes are delayed about 15 minutes: fine for levels and put/call, not for fill prices |
| Open interest by strike | Alpaca contracts list (previous close), Yahoo as backup | Saved once a day |
| Put/call volume and premium, per expiry and through the day | Computed from the snapshots | Intraday history only builds from the day collection starts |
| GEX: call wall, put wall, zero-gamma flip | Computed from open interest and IV | A model: real dealer positions aren't public |
| Key levels | $10 and $5 round numbers, prior day high/low/close, today's open, GEX walls, top open interest strikes | Levels within 0.1% merge and keep every reason |
| CPI, jobs report, FOMC days | Official BLS and Fed dates, 2021 to 2027, in `zerodte/calendar.py` | |

Not covered by free data: option price and open interest history before collection starts, and
the $200K sweep filter (needs every option trade). Those wait for ThetaData if the free tests earn it.

```
python -m agent.stan.zerodte.collect backfill    # one time: QQQ minute bars since 2021 (takes a while)
python -m agent.stan.zerodte.collect snapshot    # one snapshot now
python -m agent.stan.zerodte.collect levels      # today's levels, GEX and put/call table
python -m agent.stan.zerodte.collect status      # what has been collected
```

`schedule_trading.bat` now also adds a "Stan 0DTE data" task that runs the collector every
weekday from 9:25 AM to the close (log: `data\collect.log`). Nothing in Phase 2 places orders.

Next: Phase 3, the backtest with realistic option fills, testing each level and trigger alone.

## Notes

- Free Alpaca data: stock bars come from the IEX feed (history starts around 2016),
  and option quotes come from the indicative feed (delayed about 15 minutes).
- The snapshot job keeps options within 60 days of expiry and 15% of the stock
  price. Both are adjustable in `.env`.
- Leave `snapshot_chains` running on a machine that stays on during market hours.

## Next phases

1. Intraday study and stronger signal models
2. Risk engine (kill switch at $40, per-trade limits, limit-order walking)
3. Paper trading loop and the Expo iPhone app

Not financial advice. Options can lose 100% of the premium paid.
