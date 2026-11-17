# Stan on Discord: design

Date: 2026-11-16
Status: approved in conversation, waiting for review of this written spec

## Goal

Zy talks to Stan in plain language in Discord instead of running terminal commands. Stan answers
questions about his paper account, trades, studies and market levels, and posts short reports on
his own after his scheduled jobs.

### Success looks like

1. Zy sends Stan a DM such as "how did today go?" or "what did you learn last night?" and gets a
   short, correct answer built from real data.
2. After each scheduled job, a report arrives in Discord with no action from Zy.
3. Day-to-day use needs no terminal.

### Decisions made

| Question | Decision |
|---|---|
| Where to chat | Discord, direct messages only |
| What Stan may do | Talk and report only. No trading, pausing, tier changes or study runs from Discord |
| AI model | Claude Sonnet 5.5 (`claude-sonnet-5-5`) |
| Proactive posts | Yes: after the 9:35 AM trade, after the 6 PM study term, when a job fails, and when a job did not run |
| Architecture | A bot process on Zy's PC that reads Stan's data directly (approach A) |

### Assumptions

- Only Zy can talk to Stan. Messages from any other Discord user are ignored.
- The bot runs on the same Windows PC as everything else and starts when Zy logs in.
- Stan reads the same data the web app uses. Nothing on the PC is opened to the internet: the bot
  connects out to Discord.
- Stan keeps the voice from `agent/stan/persona.py` and stays honest about bad results.

## Components

A new package, `agent/discord/`, with four modules. Each can be tested on its own.

| Module | Purpose | Depends on |
|---|---|---|
| `tools.py` | Stan's read-only tools: plain functions that return small JSON-safe dictionaries | `Journal`, `PaperBook` and `paper.snapshot`, the broker's read methods, `zerodte.collect.today_view`, log files, Windows Task Scheduler (read only) |
| `brain.py` | Turns a message plus recent history into Stan's reply, using Claude with the tools | `tools.py`, `anthropic`, `persona.py` |
| `bot.py` | Discord connection: owner check, typing indicator, reply splitting, missed-run checks | `brain.py`, `notify.py`, `discord.py` |
| `notify.py` | `python -m agent.discord.notify <job> <exit code>`: sends one report through Discord's REST API, then exits | `tools.py`, `httpx` |

A shared `config` piece reads three new `.env` settings: `DISCORD_BOT_TOKEN`, `DISCORD_OWNER_ID`
and `ANTHROPIC_API_KEY`. Like the Alpaca keys, they are never printed or logged.

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

1. Zy sends Stan a DM. `bot.py` drops any message whose author ID is not `DISCORD_OWNER_ID`, any
   message from a bot, and any message that is not a DM.
2. The bot shows "Stan is typing..." and calls `brain.reply(history, message)`. That call runs in a
   worker thread, so the Discord connection stays alive while the model thinks.
3. `brain.py` sends Claude:
   - a system prompt (persona plus rules, below), cached;
   - the tool definitions, cached and always in the same order;
   - the last 20 messages of this conversation.

   Request settings: model `claude-sonnet-5-5`, effort `low` (this is chat), default thinking,
   `tool_choice` `auto`, and at most 8 tool calls per reply. After the 8th, Stan answers with what
   he has.
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
don't use the AI model, so they cost nothing, can't misstate a number, and still arrive if the
Anthropic API is down.

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

`notify.py` sends DMs through Discord's REST API with the bot token: it opens a DM channel with
the owner, then posts the message. It never opens the realtime Discord connection, so it works
whether or not `bot.py` is running.

## Errors

| Situation | What Stan does |
|---|---|
| Anthropic API error, rate limit or out of credit | Replies "I can't think right now:" followed by the error type. No crash. |
| The model stops because of a refusal or `max_tokens` | Replies with what it has, or a short "I couldn't finish that answer." |
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

1. In the Discord Developer Portal, create an application, add a bot, and copy its token. No
   privileged intents are needed, because Discord sends a bot the text of its own DMs without the
   Message Content intent.
2. Create a private server and invite the bot to it. Discord lets a bot DM only people it shares a
   server with.
3. In Discord, turn on Developer Mode, then copy your user ID.
4. On console.anthropic.com, create an API key and set a monthly spending limit (suggested: $5).
5. Add `DISCORD_BOT_TOKEN`, `DISCORD_OWNER_ID` and `ANTHROPIC_API_KEY` to `.env`. The
   `.env.example` file gets the three names with empty values.

`requirements.txt` gains `discord.py` and `anthropic`.

## Testing

All of these run offline as part of `python -m pytest`, with no Discord, Anthropic or Alpaca
access.

| Area | Test |
|---|---|
| Tools | Each tool returns the expected fields from a temporary journal and paper book and a fake broker. |
| Read-only guarantee | A test fails if `agent/discord/` imports order, pause, resume or training functions. |
| Brain | A fake Claude client returns scripted tool calls. Checks that the loop runs tools and returns the final text, stops at 8 tool calls, turns a tool error into a failed tool result, and turns an API error into the "can't think" reply. |
| Splitting | Long replies split into chunks of at most 2,000 characters, on line breaks where possible. |
| Owner check | Messages from other users, from bots, or outside DMs get no reply. |
| Reports | Fixed wording for: trade made, no trade, study done, job failed, settings missing. |
| Missed runs | With a log that has no entry for today, the check reports the missing run. With an entry, it says nothing. |

Manual checks:

- `python -m agent.discord.brain "how did today go?"` asks Stan one question from a terminal,
  without Discord. It needs `ANTHROPIC_API_KEY` and makes one real API call.
- After setup, start the bot and send Stan a DM.

## Out of scope

- Any action from Discord that changes something: trading, pausing, tier changes, study runs.
- More than one user, or channels in a server.
- Remembering conversations across bot restarts.
- Voice or images.
