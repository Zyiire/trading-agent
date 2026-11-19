"""The box breakout, live on the Alpaca paper account (QQQ, whole shares, one bracket order a day).

    python -m agent.stan.box.live watch [--dry-run]     # 9:31 to 10:31: watch for the breakout, send the bracket
    python -m agent.stan.box.live flatten [--dry-run]   # 3:55 PM: cancel the bracket's legs, close, record the R

The rules are the exam's (agent.stan.box.rules), so live trading and the report card can't drift apart.
Today's minutes come from Alpaca's IEX feed (the free SIP feed is about 15 minutes behind); the
15-minute trend history comes from the SIP bars in data/box_bars.db, topped up at the start of the run.

State lives in paper meta: `box_today` (today's box, signal, order and result) and `box_results`
(one row a closed trade, with its R). Every decision is written to the paper log.
Paper only: the broker refuses a live account and so does this module.
"""
from __future__ import annotations

import argparse
import math
import sys
import time as _time
from datetime import date, datetime, time, timedelta
from typing import Callable

import pandas as pd

from agent.stan.box import rules
from agent.stan.paper import PaperBook
from agent.stan.zerodte.bars import ET

SYMBOL = "QQQ"
PREFIX = "stan-box-"
WATCH_END = time(10, 31)
NO_BOX_BY = time(9, 35)            # with no bars at all by now (a holiday), the box can't be built
MAX_LATE = timedelta(minutes=3)    # enter only this soon after the breakout bar closed
WEEKLY_LIMIT_R = -3.0
MIN_TREND_BARS = 200               # EMA(200) needs at least this many earlier 15-minute bars
RISK_PCT = 0.01
FINAL = {"sent", "no_box", "no_signal", "blocked", "skipped", "failed", "closed"}
OPEN = {"sent", "flatten_failed"}  # a trade that may still hold shares
TIME_NOTE = ("The exam's 3:55 exit is the close of the 3:55 bar; this flatten runs at 3:55, about a minute "
             "before that.")


def order_id(day: date) -> str:
    """One id a day, so Alpaca itself refuses a second entry."""
    return f"{PREFIX}{day:%Y%m%d}"


def _t(x: datetime) -> time:
    return x.astimezone(ET).time()


def weekly_r(book: PaperBook, day: date) -> float:
    """Sum of closed box trades' R in the Monday-to-Friday week of `day`."""
    week = day.isocalendar()[:2]
    return sum(r["r"] for r in book.get("box_results", []) or []
               if r.get("r") is not None and date.fromisoformat(r["day"]).isocalendar()[:2] == week)


def _sign(state: dict) -> int:
    return 1 if state["order"]["side"] == "buy" else -1


# ---------------------------------------------------------------- watch: one pass
def step(broker, book: PaperBook, hist15: pd.DataFrame, today: pd.DataFrame, now: datetime,
         dry_run: bool = False, log: Callable[[str], None] = print, sleep=_time.sleep) -> bool:
    """One look at the market. True once today is decided (order sent, or no trade today).

    `hist15`: finished 15-minute bars before 9:00 today (SIP). `today`: today's 1-minute bars from
    9:00 on (IEX), ET-indexed o/h/l/c. `now`: the current time, tz-aware.
    """
    now = now.astimezone(ET)
    d = now.date()
    day = d.isoformat()
    state = book.get("box_today")

    def say(kind: str, text: str) -> None:
        log(text)
        if not dry_run:
            book.log(day, kind, text)

    def decide(status: str, reason: str, kind: str = "thought", **extra) -> bool:
        say(kind, f"Box breakout: no trade today, {reason}.")
        if not dry_run:
            book.set("box_today", {"day": day, "status": status, "reason": reason, **extra})
        return True

    if state and state.get("day") == day and state.get("status") in FINAL:
        return True
    if state and state.get("day") != day and state.get("status") in OPEN:
        say("fail", f"The box trade from {state['day']} is still open; closing it before today's watch.")
        state = flatten(broker, book, now, dry_run=dry_run, log=log, sleep=sleep)
        if dry_run or (state or {}).get("status") in OPEN:
            say("fail", "Box breakout: no new trade while an earlier box trade is still open.")
            return True

    cid = order_id(d)
    existing = broker.get_order(cid)
    if existing is not None:
        say("thought", f"Today's box order {cid} is already with Alpaca ({existing.status}); not sending another.")
        known = state and state.get("day") == day and state.get("status") in OPEN
        if not dry_run and not known:      # never replace a record that knows more (close orders, results)
            book.set("box_today", _recovered(broker, cid, day, existing))
        return True

    if book.get("halted", False):
        return decide("skipped", "trading is paused (" + (book.get("halt_reason") or "switched off") + ")")
    strategy = book.get("strategy", "selloff")
    if strategy != "box":
        return decide("skipped", f"the paper account trades {strategy}, not the box breakout")
    wk = weekly_r(book, d)
    if wk <= WEEKLY_LIMIT_R:
        return decide("skipped", f"this week's box trades have lost {wk:+.2f}R, at or past the 3R weekly limit")
    close = broker.get_clock().next_close.astimezone(ET)
    if close.date() == d and close.time() < time(16, 0):
        return decide("skipped", f"half day: the bracket would expire at the early close ({close:%H:%M}), "
                                 "leaving the position without a stop")

    # Only count bars whose last minute has arrived, so a late minute can't fake a bar's close.
    cutoff = now
    if len(today):
        cutoff = min(now, today.index.max().to_pydatetime() + timedelta(minutes=1))
    five = rules.bars(today, 5, now=cutoff)
    bx = rules.box(five)
    if bx is None:
        if (len(today) and _t(cutoff) >= time(9, 30)) or _t(now) >= NO_BOX_BY:
            got = len(five[(five.index.hour == 9) & (five.index.minute < 30)])
            return decide("no_box", f"the 9:00-9:30 box is incomplete ({got} of 6 five-minute bars; "
                                    "a holiday, a half day or a data gap)")
        return False
    raw = rules.signal(five, bx, None)
    if raw is None:
        if _t(cutoff) >= time(10, 30):
            return decide("no_signal", f"nothing closed outside the box (${bx[0]:.2f} high, ${bx[1]:.2f} low) "
                                       "by 10:30", box=list(bx))
        return False

    closed_at = raw.bar_open + pd.Timedelta(minutes=5)
    side_word = "above the high" if raw.side == "long" else "below the low"
    seen = f"the {raw.bar_open:%H:%M} bar closed at ${raw.entry:.2f}, {side_word} of the box"
    if now - closed_at.to_pydatetime() > MAX_LATE:
        return decide("skipped", f"{seen}, but I only saw it at {now:%H:%M}, too late to enter at its price",
                      box=list(bx))
    fifteen = pd.concat([hist15, rules.bars(today, 15, now=cutoff)])
    if len(hist15) < MIN_TREND_BARS:
        return decide("skipped", f"{seen}, but there are only {len(hist15)} 15-minute bars of history "
                                 f"for the trend filter (it needs {MIN_TREND_BARS})", box=list(bx))
    sig = rules.signal(five, bx, lambda t: rules.trend_up(fifteen, t))
    if sig is None:
        up = rules.trend_up(fifteen, closed_at)
        trend = "unknown" if up is None else ("up" if up else "down")
        return decide("blocked", f"{seen}, but the 15-minute trend is {trend}, so the trend filter blocks a "
                                 f"{raw.side}", box=list(bx))

    account = broker.get_account()
    if not account.is_paper:
        raise RuntimeError("Stan only trades the paper account.")
    qty = rules.size(account.equity, sig.risk, account.buying_power, sig.entry, RISK_PCT)
    signal = {"side": sig.side, "bar_open": sig.bar_open.isoformat(), "entry": sig.entry, "stop": sig.stop,
              "target": sig.target, "risk": sig.risk}
    if qty == 0:
        want = math.floor(account.equity * RISK_PCT / sig.risk) if sig.risk > 0 else 0
        if want < 1:
            why = (f"1% of ${account.equity:,.2f} is less than one share's risk of ${sig.risk:.2f}")
        else:
            why = (f"{want} shares (1% risk at ${sig.risk:.2f} a share) would cost ${want * sig.entry:,.2f}, "
                   f"more than 95% of the ${account.buying_power:,.2f} buying power")
        return decide("skipped", f"{seen}, but {why}", box=list(bx), signal=signal)

    side = "buy" if sig.side == "long" else "sell"
    what = (f"Box breakout: {seen}. Sending a bracket order to {'buy' if side == 'buy' else 'short'} {qty} "
            f"{SYMBOL} at market, stop ${sig.stop:.2f}, target ${sig.target:.2f} (3R), risking about "
            f"${qty * sig.risk:,.2f} (1% of ${account.equity:,.2f}). Order id {cid}.")
    if dry_run:
        say("thought", what + " (dry run, nothing sent)")
        return True
    try:
        broker.submit_bracket_order(SYMBOL, qty, side, sig.stop, sig.target, cid)
    except Exception as err:
        try:
            taken = broker.get_order(cid) is not None
        except Exception:
            taken = False
        if not taken:
            return decide("failed", f"{seen}, but the bracket order failed: {type(err).__name__}: {err}",
                          kind="fail", box=list(bx), signal=signal)
    book.set("box_today", {"day": day, "status": "sent", "reason": seen, "box": list(bx), "signal": signal,
                           "order": {"id": cid, "qty": qty, "side": side, "stop": sig.stop, "target": sig.target,
                                     "risk": sig.risk, "equity": account.equity}})
    say("progress", what)
    return True


def watch(broker, book: PaperBook, hist15: pd.DataFrame, fetch_today: Callable[[datetime], pd.DataFrame],
          clock: Callable[[], datetime] = lambda: datetime.now(ET), sleep=_time.sleep, dry_run: bool = False,
          log: Callable[[str], None] = print) -> dict | None:
    """Once a minute until 10:31: fetch today's minutes and take one step. Stops early once decided.

    A failed fetch or step (a broker hiccup) is logged and retried the next minute; the Alpaca order-id
    check at the top of `step` makes a retry safe against a second entry."""
    problem = None
    while True:
        now = clock()
        decided = False
        try:
            today = fetch_today(now)
            decided = step(broker, book, hist15, today, now, dry_run=dry_run, log=log, sleep=sleep)
        except Exception as err:
            problem = f"{type(err).__name__}: {err}"
            log(f"Box watch problem at {now.astimezone(ET):%H:%M}: {problem}. Trying again next minute.")
        if decided:
            break
        if _t(now) >= WATCH_END:
            day = now.astimezone(ET).date().isoformat()
            state = book.get("box_today")
            if not (state and state.get("day") == day):
                why = "the watch ended at 10:31 without a decision" + (f" (last problem: {problem})" if problem else "")
                log(f"Box breakout: no trade today, {why}.")
                if not dry_run:
                    book.log(day, "fail", f"Box breakout: no trade today, {why}.")
                    book.set("box_today", {"day": day, "status": "failed", "reason": why})
            break
        sleep(max(1.0, 65 - now.second - now.microsecond / 1e6))   # wake about 5 s into the next minute
    return book.get("box_today")


def _recovered(broker, cid: str, day: str, existing) -> dict:
    """Today's record rebuilt from Alpaca (after a crash before it was saved): stop and target from the legs."""
    stop = target = None
    try:
        _, legs = broker.get_bracket(cid)
    except Exception:
        legs = []
    for leg in legs:
        if leg.order_type in ("stop", "stop_limit") and leg.stop_price is not None:
            stop = leg.stop_price
        elif leg.order_type == "limit" and leg.limit_price is not None:
            target = leg.limit_price
    risk = abs(target - stop) / (1 + rules.TARGET_R) if stop is not None and target is not None else None
    return {"day": day, "status": "sent", "reason": "found at Alpaca after a restart",
            "order": {"id": cid, "qty": existing.qty, "side": existing.side, "stop": stop, "target": target,
                      "risk": risk}}


# ---------------------------------------------------------------- flatten
def _fills(broker, state: dict):
    """(entry order, exit fills [(qty, price, reason)], shares still held) from Alpaca's orders."""
    parent, legs = broker.get_bracket(state["order"]["id"])
    fills = []
    for leg in legs:
        if leg.filled_qty > 0:
            kind = {"limit": "target", "stop": "stop", "stop_limit": "stop"}.get(leg.order_type or "")
            if kind is None:
                target, stop = state["order"].get("target"), state["order"].get("stop")
                kind = ("target" if target is not None and stop is not None
                        and abs(leg.filled_avg_price - target) < abs(leg.filled_avg_price - stop) else "stop")
            fills.append((leg.filled_qty, leg.filled_avg_price, kind))
    for cid in state.get("close_orders", []):
        o = broker.get_order(cid)
        if o is not None and o.filled_qty > 0:
            fills.append((o.filled_qty, o.filled_avg_price, "time"))
    entered = parent.filled_qty if parent is not None else 0.0
    left = entered - sum(q for q, _, _ in fills)
    return parent, fills, left


def held_qty(broker, book: PaperBook) -> float:
    """Shares the box trade still holds (negative for a short), for the 9:35 reconcile. Uses Alpaca's
    fills; if those can't be read, the recorded order size."""
    state = book.get("box_today") or {}
    if state.get("status") not in OPEN or not state.get("order"):
        return 0.0
    try:
        _, _, left = _fills(broker, state)
    except Exception:
        left = float(state["order"].get("qty") or 0)
    return _sign(state) * left


def flatten(broker, book: PaperBook, now: datetime | None = None, dry_run: bool = False, sleep=_time.sleep,
            wait: int = 30, log: Callable[[str], None] = print) -> dict | None:
    """Cancel Stan's open box orders, close what the box trade still holds, and record its R from the fills."""
    now = (now or datetime.now(ET)).astimezone(ET)
    state = book.get("box_today")
    day = (state or {}).get("day") or now.date().isoformat()

    def say(kind: str, text: str) -> None:
        log(text)
        if not dry_run:
            book.log(day, kind, text)

    if not state or state.get("status") not in OPEN or not state.get("order"):
        log("Box flatten: no open box trade, nothing to close.")
        return state
    if dry_run:
        _, _, left = _fills(broker, state)
        say("thought", f"Box flatten: would cancel the open {PREFIX} orders and close {left:g} {SYMBOL} "
                       "(dry run, nothing sent).")
        return state

    broker.cancel_orders(PREFIX)
    parent, fills, left = _fills(broker, state)
    if left > 1e-9:
        n = len(state.get("close_orders", [])) + 1
        cid = f"{state['order']['id']}-flat{n}"
        state = {**state, "close_orders": [*state.get("close_orders", []), cid]}
        book.set("box_today", state)
        close_side = "sell" if state["order"]["side"] == "buy" else "buy"    # from the box trade, not the account
        order, err = None, None
        for attempt in range(3):
            if attempt:                 # a leg may have filled meanwhile: never close more than is left
                parent, fills, left = _fills(broker, state)
                if left <= 1e-9:
                    break
            try:
                order = broker.close_position(SYMBOL, left, close_side, cid)
                err = None
                break
            except Exception as e:      # the cancelled legs may still be holding the shares for a moment
                err = e
                taken = broker.get_order(cid)
                if taken is not None:
                    order, err = taken, None
                    break
                sleep(2)
        if left > 1e-9:
            if order is None:
                why = f"{type(err).__name__}: {err}" if err else "no order came back"
                return _failed(book, state, say, f"couldn't send the closing order ({why})")
            for _ in range(wait):
                o = broker.get_order(cid)
                if o is not None and o.filled_qty >= left - 1e-9:
                    break
                sleep(1)
            parent, fills, left = _fills(broker, state)
            if left > 1e-9:
                return _failed(book, state, say, f"the closing order {cid} hasn't filled yet ({left:g} shares left)")

    day_ = state["day"]
    if parent is None or parent.filled_qty <= 0:
        result = {"day": day_, "side": state["order"]["side"], "qty": 0, "entry": None, "exit": None,
                  "r": None, "pnl": 0.0, "reason": "entry not filled"}
        say("thought", f"Box flatten: the entry order was {parent.status if parent else 'not found'}; "
                       "nothing was bought or shorted.")
    else:
        sign, qty, entry = _sign(state), parent.filled_qty, parent.filled_avg_price
        exit_px = sum(q * p for q, p, _ in fills) / sum(q for q, _, _ in fills)
        reasons = {k for _, _, k in fills}
        reason = "time" if "time" in reasons else ("stop" if "stop" in reasons else "target")
        if reason == "time" and (now.date() != date.fromisoformat(day_) or _t(now) < time(15, 50)):
            reason = "early"            # the next morning's cleanup, or the kill switch
        risk = state["order"].get("risk") or abs(entry - (state["order"].get("stop") or entry)) or None
        r = round(sign * (exit_px - entry) / risk, 4) if risk else None
        pnl = round(sign * (exit_px - entry) * qty, 2)
        result = {"day": day_, "side": state["order"]["side"], "qty": qty, "entry": entry,
                  "exit": round(exit_px, 4), "r": r, "pnl": pnl, "reason": reason}
        how = {"time": f"closed at 3:55 at ${exit_px:.2f}. {TIME_NOTE}",
               "early": f"closed early, on {now:%b} {now.day} at {now:%H:%M}, at ${exit_px:.2f}.",
               "stop": f"the stop filled at ${exit_px:.2f}", "target": f"the 3R target filled at ${exit_px:.2f}"}
        say("win" if pnl > 0 else "fail",
            f"Box trade done: {'long' if sign > 0 else 'short'} {qty:g} {SYMBOL} from ${entry:.2f}, {how[reason]}"
            + (f" Result {r:+.2f}R" if r is not None else " Result") + f" ({'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}).")
    state = {**state, "status": "closed", "result": result}
    book.set("box_today", state)
    book.set("box_results", [*(book.get("box_results", []) or []), result])
    return state


def _failed(book: PaperBook, state: dict, say, why: str) -> dict:
    state = {**state, "status": "flatten_failed", "reason": why}
    book.set("box_today", state)
    say("fail", f"Box flatten problem: {why}. The position may still be open; the next run tries again.")
    return state


# ---------------------------------------------------------------- trend history
TOP_UP_DAYS = 30


def _last_stored(db) -> datetime | None:
    import sqlite3
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            ts = con.execute("SELECT MAX(ts) FROM bars_1m WHERE symbol=?", (SYMBOL,)).fetchone()[0]
    except sqlite3.Error:
        return None
    return None if ts is None else pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(ET).to_pydatetime()


def trend_history(client, db, start: datetime, backfill=None, load=None, log: Callable[[str], None] = print):
    """Finished 15-minute SIP bars before `start` (9:00 today), after a quick top-up of the stored bars.

    The top-up only runs when the stored bars are at most 30 days behind, so a missing or stale database
    can't hold up the watch. Without recent bars the history is empty and the watch skips the day."""
    from agent.stan.box import data
    backfill, load = backfill or data.backfill, load or data.load
    empty = pd.DataFrame(columns=["o", "h", "l", "c"], index=pd.DatetimeIndex([], tz=ET), dtype=float)
    last = _last_stored(db)
    if last is None or start - last > timedelta(days=TOP_UP_DAYS):
        log(f"The SIP bars in {db} " + (f"end {last:%Y-%m-%d}" if last else "are missing")
            + f", more than {TOP_UP_DAYS} days behind; not fetching them during the watch. "
            "Run python -m agent.stan.box.data backfill first. No trend history today.")
        return empty
    try:
        n = backfill(client, db, log=log)
        log(f"Topped up {db} with {n:,} SIP bars.")
    except Exception as err:
        log(f"Couldn't top up the SIP bars ({type(err).__name__}: {err}); using what's stored.")
    try:
        stored = load(db)
    except Exception as err:
        log(f"Couldn't read {db} ({type(err).__name__}: {err}); no trend history.")
        return empty
    return rules.bars(stored[stored.index < pd.Timestamp(start)], 15)


# ---------------------------------------------------------------- command line
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m agent.stan.box.live", description="The box breakout on QQQ.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("watch", "9:31-10:31: watch for the breakout and send one bracket order"),
                        ("flatten", "3:55 PM: cancel the box orders, close the position, record the R")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--dry-run", action="store_true", help="show what Stan would do; send nothing")
        p.add_argument("--journal", default="data/stan.db")
        p.add_argument("--db", default=None, help="QQQ minute bars (default data/box_bars.db)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from agent.broker.alpaca import AlpacaBroker
    from agent.config import ConfigError, load_settings
    from agent.stan.box import data

    try:
        settings = load_settings()
    except ConfigError as err:
        print(f"Setup problem: {err}")
        return 1
    broker = AlpacaBroker(settings)
    book = PaperBook(args.journal)
    try:
        if args.cmd == "flatten":
            state = flatten(broker, book, dry_run=args.dry_run)
            return 1 if (state or {}).get("status") == "flatten_failed" else 0

        from alpaca.data.historical import StockHistoricalDataClient
        from agent.stan.zerodte.bars import fetch_alpaca

        db = args.db or data.DB_PATH
        client = StockHistoricalDataClient(settings.api_key, settings.secret_key)
        start = datetime.combine(datetime.now(ET).date(), time(9, 0), tzinfo=ET)
        hist15 = trend_history(client, db, start)
        last = hist15.index.max() if len(hist15) else None
        print(f"Trend history: {len(hist15):,} 15-minute bars from the SIP feed"
              + (f", last one {last:%Y-%m-%d %H:%M}." if last is not None else "."))
        print("Today's minutes from 9:00 come from the IEX feed (SIP is about 15 minutes behind).")

        def fetch_today(now: datetime) -> pd.DataFrame:
            return rules.minute_frame(fetch_alpaca(client, SYMBOL, start, now, feed="iex"))

        state = watch(broker, book, hist15, fetch_today, dry_run=args.dry_run)
        print(f"Box watch done: {(state or {}).get('status', 'nothing recorded')}"
              + (" (dry run, nothing sent or saved)" if args.dry_run else "") + ".")
        return 0
    finally:
        book.close()


if __name__ == "__main__":
    sys.exit(main())
