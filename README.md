# Stan: AI Trading Agent

Stan is a trader in training. Before he trades anything, he goes to school:
he studies up to 15 years of price history, sits an exam on every year he has
not seen, and journals what his wins and losses teach him. A phone app shows
his thinking and task progress live.

**Nothing here can trade.** There is no order code yet, the Alpaca connection is
read-only, and live mode is locked in `agent/config.py`.

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
