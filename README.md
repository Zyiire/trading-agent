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
| `agent/stan/` | Stan: persona, journal, study method, training school |
| `agent/api.py` | Read-only API for the phone app |
| `stan-app/` | Expo (React Native) iPhone app |
| `tests/` | Offline tests using fake Alpaca clients |

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
