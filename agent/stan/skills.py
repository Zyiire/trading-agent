"""Stan's skill library: what each tool in the Day Trader skill pack does, and how Stan uses it."""
from __future__ import annotations

PACK = {
    "name": "The Day Trader skill (Trader skill v2)",
    "author": "Jake Schincariol",
    "source": "https://github.com/Jakeschincariol/day-trader-skill",
    "version": "3d0f4ef (2026-10-20)",
    "license": "MIT",
}

# status: "in_use" (Stan runs it), "learned" (in his library, not wired into decisions yet),
# "locked" (deliberately off until Stan passes the go-live checks).
SKILLS = [
    {"key": "trader-scanner", "name": "Chart Scanner", "status": "in_use",
     "job": "Scans charts for setups: opening-range breakouts, flags, oversold bounces.",
     "stan": "Its opening-range breakout rule is the setup in Stan's day-trading exam and nightly practice. "
             "Its oversold-bounce scan is a cousin of his selloff bounce."},
    {"key": "trader-entry", "name": "Entry Timer", "status": "in_use",
     "job": "Calls the entry on the first 5-minute bar that closes above the opening-range high.",
     "stan": "Stan's trigger: a close above the range, never a wick, never a bar still forming."},
    {"key": "trader-exit", "name": "Exit Planner", "status": "in_use",
     "job": "Sets the stop and target before the entry.",
     "stan": "Stop at the range midpoint, target 1R (2R also tested), out by 3:55 PM win or lose."},
    {"key": "trader-timemachine", "name": "Time Machine", "status": "in_use",
     "job": "Replays the rest of the day 10,000 times and re-runs the rule on 60 days of real sessions.",
     "stan": "Runs before every practice trade, using only earlier sessions. Stan then checks whether its "
             "odds came true, so he learns how far to trust it. Its history replay is his day-trading exam."},
    {"key": "trader-budget", "name": "Budget Manager", "status": "in_use",
     "job": "Sizes a trade to the budget and the max loss per trade, in whole shares.",
     "stan": "Sizes each practice trade for the $50 plan with a $5 max loss. It refuses most stocks, "
             "because one share costs more than the plan allows."},
    {"key": "trader-risk", "name": "Risk Manager", "status": "in_use",
     "job": "Checks every order against the limits and blocks anything over them.",
     "stan": "Checks each sized practice trade, then Stan applies the day limits: 3 trades and $10 of "
             "risk per day."},
    {"key": "trader-broker", "name": "Order Executor", "status": "locked",
     "job": "Sends approved orders to the broker, only after a risk PASS and the user's yes.",
     "stan": "Locked. Stan can't place orders until he passes the go-live checks."},
    {"key": "trader-alerts", "name": "Price Alerts", "status": "learned",
     "job": "Rings on a price, a stop, the target or the 3:55 flat time.",
     "stan": "Not needed while Stan only practices. It will watch stops and the flat time once paper "
             "trading starts."},
    {"key": "trader-news", "name": "News Monitor", "status": "learned",
     "job": "Tracks headlines and SEC 8-K filings for each stock and flags big moves.",
     "stan": "Not wired in yet. Next step: tag practice trades that had news, to learn whether news days "
             "behave differently."},
    {"key": "trader-analyst", "name": "Earnings Analyst", "status": "learned",
     "job": "Reads filed earnings numbers from SEC EDGAR and sums up what changed.",
     "stan": "Not wired in yet. Useful later to skip day trades on earnings days."},
    {"key": "trader-pelosi", "name": "Pelosi Tracker", "status": "learned",
     "job": "Reads Congress members' stock-trade disclosures as they are filed.",
     "stan": "Reference only. Filings arrive up to 45 days after the trade, far too late for day trading."},
]


def library() -> dict:
    counts = {s: sum(1 for k in SKILLS if k["status"] == s) for s in ("in_use", "learned", "locked")}
    return {"pack": PACK, "skills": SKILLS, "counts": counts}
