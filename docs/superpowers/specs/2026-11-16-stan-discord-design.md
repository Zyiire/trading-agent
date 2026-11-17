# Stan on Discord: design

Date: 2026-11-16
Status: approved in conversation, waiting for review of this written spec

## Goal

Zy talks to Stan in plain language in Discord instead of running terminal commands. Stan answers
questions about his paper account, trades, studies and market levels, and posts short reports on
his own after his scheduled jobs.

### Success looks like

1. Zy writes in Stan's channel, for example "how did today go?" or "what did you learn last
   night?", and gets a short, correct answer built from real data.
2. After each scheduled job, a report arrives in Discord with no action from Zy.
3. Day-to-day use needs no terminal.

### Decisions made

| Question | Decision |
|---|---|
| Where to chat | Discord: one channel (for example `#stan`) in Zy's private server. Stan doesn't use DMs |
| What Stan may do | Talk and report only. No trading, pausing, tier changes or study runs from Discord |
| AI model | Free and local: a tool-calling model in Ollama on Zy's GPU (RTX 3060, 12 GB). Default `qwen3:8b`, changeable with `STAN_MODEL`. No API key, no cost per message, nothing leaves the PC |
| Proactive posts | Yes: after the 9:35 AM trade, after the 6 PM study term, when a job fails, and when a job did not run |
| Architecture | A bot process on Zy's PC that reads Stan's data directly (approach A) |

### Assumptions

- Only Zy can talk to Stan, and only in his channel. Messages from any other user, from bots, or
  from any other channel are ignored.
- The bot runs on the same Windows PC as everything else and starts when Zy logs in.
- Stan reads the same data the web app uses. Nothing on the PC is opened to the internet: the bot
  connects out to Discord.
- Stan keeps the voice from `agent/stan/persona.py` and stays honest about bad results.

## Components

A new package, `agent/discord/`, with four modules. Each can be tested on its own.

| Module | Purpose | Depends on |
|---|---|---|
| `tools.py` | Stan's read-only tools: plain functions that return small JSON-safe dictionaries | `Journal`, `PaperBook` and `paper.snapshot`, the broker's read methods, `zerodte.collect.today_view`, log files, Windows Task Scheduler (read only) |
| `brain.py` | Turns a message plus recent history into Stan's reply, using the local model with the tools | `tools.py`, Ollama's chat API at `OLLAMA_URL` (called with `httpx`), `persona.py` |
| `bot.py` | Discord connection: owner check, typing indicator, reply splitting, missed-run checks | `brain.py`, `notify.py`, `discord.py` |
| `notify.py` | `python -m agent.discord.notify <job> <exit code>`: sends one report through Discord's REST API, then exits | `tools.py`, `httpx` |

A shared `config` piece reads the new `.env` settings:

| Setting | Required | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Yes | none |
| `DISCORD_OWNER_ID` | Yes | none |
| `DISCORD_CHANNEL_ID` | Yes | none |
| `STAN_MODEL` | No | `qwen3:8b` |
| `OLLAMA_URL` | No | `http://127.0.0.1:11434` |

Like the Alpaca keys, the bot token is never printed or logged.

### Tools

Every tool is read-only. None of them imports or calls order, pause, resume, tier-change or
training code.

| Tool | Returns | Built on |
|---|---|---|
| `account` | Equity, cash, day change, positions, selected tier, whether trading is paused and why | Broker `get_account` and `get_positions`, `PaperBook` meta |
| `recent_trades` | The last N orders and trades with status, side, quantity and price | `PaperBook.trades()` |
| `trading_log` | Stan's notes from recent runs ("no trade needed...") | `PaperBook.logs()` |
| `study_report` | Date of the last term, verdict, report card, comparison with simply holding | `Journal.meta()` and `Journal.thoughts()` |
| `exams` | Year-by-year exam results | `Journal.exams()` |
| `lessons` | His lessons (mistakes and strengths) | `Journal.lessons()` |
| `strategy` | Tier rankings and the strategy currently picked | `Journal.meta()` (arena results), `PaperBook` meta |
| `levels` | 0DTE QQQ levels, GEX and the put/call table from the latest snapshot | `zerodte.collect.today_view` |
| `schedule` | For each Stan task: last run, last result, next run, plus the last lines and exit code of its log | `Get-ScheduledTask` output, `data\*.log` |
| `market_clock` | Whether the market is open, and the next open and close | Broker `get_clock` |

Tools take only simple arguments, such as a `limit` for list tools or a `day` for `levels`. Results
are trimmed (for example, at most 20 trades) so one answer can't flood the model's context.

## How a conversation works

1. Zy writes in Stan's channel. No @mention is needed. `bot.py` drops any message whose author ID
   is not `DISCORD_OWNER_ID`, any message from a bot, and any message outside
   `DISCORD_CHANNEL_ID`.
2. The bot shows "Stan is typing..." and calls `brain.reply(history, message)`. That call runs in a
   worker thread, so the Discord connection stays alive while the model thinks.
3. `brain.py` sends the local model, through Ollama's `/api/chat` endpoint:
   - a system prompt (persona plus rules, below);
   - the tool definitions;
   - the last 20 messages of this conversation.

   Request settings: model `STAN_MODEL`, `stream: false`, the model's thinking mode turned off
   (`think: false`) for faster replies, and at most 8 tool calls per reply. After the 8th, Stan
   answers with what he has.

   A small local model is easier to confuse than a large hosted one, so tool results stay short:
   only the fields that answer the question, numbers already rounded, and dates in plain form.
4. Tool calls run locally. A tool that raises an exception returns its error to the model as a
   failed tool result, so Stan can say which data he couldn't reach.
5. The bot sends the reply, split into chunks under Discord's 2,000-character limit, on line
   breaks where possible.
6. History is kept in memory only (the last 20 messages). Restarting the bot starts a new
   conversation. Stan's lasting memory is his journal and trades, which tools always read fresh.

### System prompt rules

- You are Stan, using the name, goal, bio and principles from `persona.py`.
- Only state numbers and facts that a tool returned in this conversation. If no tool covers the
  question, say you don't know.
- Keep replies short and in plain language. Use a short list only for several items.
- Be honest when results are bad, as your journal is.
- You can't trade, pause, change tier or start a study term from Discord. If asked, say so and
  point to the web app (`stan.bat`) or the automatic schedule.
- Paper money only. Never suggest that live trading is on.

## Daily reports

Reports are built from the numbers with fixed wording, the same way journal entries are. They
don't use the AI model, so they can't misstate a number and still arrive when Ollama isn't
running.

| Trigger | Sent by | Content |
|---|---|---|
| End of `run_open.bat` | `notify open <code>` | Exit code 0: what was bought or sold today (symbol, side, quantity), or Stan's note on why nothing traded, plus equity and the day's change. Any other code: a failure report. |
| End of `run_study.bat` | `notify study <code>` | Exit code 0: the verdict line, the report card ($1 with Stan compared with $1 simply held), and the newest lesson. Any other code: a failure report. |
| End of `run_collect.bat` | `notify collect <code>` | Only when the exit code isn't 0: a failure report. |
| A job didn't run | `bot.py` timer | On weekdays at 9:50 AM and 6:30 PM New York time: if `data\open.log` or `data\study.log` has no entry for today, Stan says that run is missing and that the PC was probably off or asleep. |

A failure report names the job and its exit code, and includes the last 10 lines of its log.

Each `.bat` job saves its exit code right after its Python run, then uses the saved value for
both the log line and the report:

```bat
%PY% -m agent.stan.paper >> data\open.log 2>&1
set "CODE=%errorlevel%"
echo Exit code: %CODE% >> data\open.log
%PY% -m agent.discord.notify open %CODE% >> data\open.log 2>&1
```

This way the report carries the job's own result, not the result of a later command. If
Discord settings are missing, `notify` writes "Discord not set up, skipping report" and exits 0.
If sending fails, it writes the error to the same log and exits 0. Either way, a report can never
make a job look failed.

All reports, including missed-run notices, go to Stan's channel. `notify.py` posts them through
Discord's REST API (`POST /channels/{DISCORD_CHANNEL_ID}/messages`) with the bot token. It never
opens the realtime Discord connection, so it works whether or not `bot.py` is running.

## Errors

| Situation | What Stan does |
|---|---|
| Ollama isn't running | Replies "My brain isn't running. Start Ollama and ask again." No crash. |
| The model isn't downloaded | Replies with the exact command to fix it: `ollama pull <STAN_MODEL>`. |
| The model returns no text, or calls a tool that doesn't exist | Unknown tools get a failed tool result. An empty answer becomes "I couldn't finish that answer. Try asking another way." |
| One tool fails (for example, Alpaca is unreachable) | That tool returns its error. Stan answers with the rest and names what he couldn't reach. |
| Discord connection drops | `discord.py` reconnects on its own. If the process exits, `run_discord.bat` restarts it after 30 seconds. |
| Settings missing from `.env` | The bot prints which setting is missing and exits. `notify` skips its report, as above. |

## Running and setup

### Keeping the bot running

- `run_discord.bat` uses the same Python lookup as the other `.bat` jobs. It runs
  `python -m agent.discord.bot` in a loop: when the bot exits, it waits 30 seconds and starts it
  again. Output goes to `data\discord.log`.
- A scheduled task, **Stan Discord**, starts `run_discord.bat` when Zy logs in. It is created with
  PowerShell (`Register-ScheduledTask`) so that it has no time limit (Windows otherwise stops tasks
  after 72 hours), keeps running on battery, and runs only one copy at a time.
- `schedule_trading.bat` creates this task along with the three existing ones.

### One-time setup for Zy (about 10 minutes)

The README gets step-by-step instructions:

1. In the Discord Developer Portal, create an application, add a bot, and copy its token. Turn
   off **Public Bot**, and turn on **Message Content Intent**. Discord requires that intent for a
   bot to read normal channel messages.
2. Create a private server and a channel in it (for example `#stan`). Invite the bot with the
   View Channels, Send Messages and Read Message History permissions.
3. In Discord, turn on Developer Mode. Copy your user ID (right-click your name, then Copy User
   ID) and the channel's ID (right-click the channel, then Copy Channel ID).
4. Download the model once: `ollama pull qwen3:8b` (about 5 GB). Ollama is already installed and
   starts with Windows.
5. Add `DISCORD_BOT_TOKEN`, `DISCORD_OWNER_ID` and `DISCORD_CHANNEL_ID` to `.env`. The
   `.env.example` file gets all five new settings, with the two optional ones commented out.

`requirements.txt` gains `discord.py`. Ollama is called with `httpx`, which is already a
dependency.

### Using the GPU

The model takes about 6 GB of the RTX 3060's 12 GB while loaded. Ollama unloads it after about
5 idle minutes, so the first answer after a quiet spell takes a few extra seconds while it loads
again. Nothing else in Stan uses the GPU.

## Testing

All of these run offline as part of `python -m pytest`, with no Discord, Ollama or Alpaca
access.

| Area | Test |
|---|---|
| Tools | Each tool returns the expected fields from a temporary journal and paper book and a fake broker. |
| Read-only guarantee | A test fails if `agent/discord/` imports order, pause, resume or training functions. |
| Brain | A fake Ollama endpoint returns scripted tool calls. Checks that the loop runs tools and returns the final text, stops at 8 tool calls, turns a tool error or an unknown tool into a failed tool result, and turns "Ollama not running" and "model not downloaded" into the right replies. |
| Splitting | Long replies split into chunks of at most 2,000 characters, on line breaks where possible. |
| Owner check | Messages from other users, from bots, or from other channels get no reply. |
| Reports | Fixed wording for: trade made, no trade, study done, job failed, settings missing. |
| Missed runs | With a log that has no entry for today, the check reports the missing run. With an entry, it says nothing. |

Manual checks:

- `python -m agent.discord.brain "how did today go?"` asks Stan one question from a terminal,
  without Discord, using the real local model.
- A short list of sample questions (account, today's trades, last report card, levels, "did
  everything run?") is asked once before going live, and each answer is checked against the
  data, to confirm the chosen model reads its tools correctly. If it doesn't, try another model
  with `STAN_MODEL`.
- After setup, start the bot and write to Stan in his channel.

## Out of scope

- Any action from Discord that changes something: trading, pausing, tier changes, study runs.
- More than one user, more than one channel, or DMs.
- Remembering conversations across bot restarts.
- Voice or images.
