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
NO_BOX_BY = time(9, 35)            # with no bars at all by now, ask the clock: a holiday, or a feed glitch
SIP_READY = time(9, 46)            # the free SIP feed has the whole 9:00-9:30 box 16 minutes after it ends
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
         dry_run: bool = False, log: Callable[[str], None] = print, sleep=_time.sleep,
         fetch_sip_box: Callable[[], pd.DataFrame] | None = None) -> bool:
    """One look at the market. True once today is decided (order sent, or no trade today).

    `hist15`: finished 15-minute bars before 9:00 today (SIP); `hist15.attrs["problem"]`, if set, says
    why the history can't be trusted today. `today`: today's 1-minute bars from 9:00 on (IEX),
    ET-indexed o/h/l/c. `now`: the current time, tz-aware. `fetch_sip_box`: today's 9:00-9:30 SIP
    minutes, used after 9:46 when an IEX box bucket is empty.
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
    clk = broker.get_clock()
    close = clk.next_close.astimezone(ET)
    if close.date() == d and close.time() < time(16, 0):
        return decide("skipped", f"half day: the bracket would expire at the early close ({close:%H:%M}), "
                                 "leaving the position without a stop")

    if not len(today):
        if _t(now) < NO_BOX_BY:
            return False                    # the first minutes may still be on their way
        if not clk.is_open:
            return decide("no_box", "the market is closed today (a holiday), so there is no box")
        log(f"Box watch: no IEX minutes for today at {now:%H:%M} although the market is open (a feed glitch); "
            "trying again next minute.")
        return False

    # Only count bars whose last minute has arrived, so a late minute can't fake a bar's close.
    cutoff = min(now, today.index.max().to_pydatetime() + timedelta(minutes=1))
    five = rules.bars(today, 5, now=cutoff)
    bx = rules.box(five)
    if bx is None:
        if _t(cutoff) < time(9, 30) and _t(now) < SIP_READY:
            return False                    # the box is still forming
        got = _box_buckets(five)
        if fetch_sip_box is None:
            return decide("no_box", f"the 9:00-9:30 box is incomplete ({got} of 6 five-minute bars; "
                                    "a half day or a data gap)")
        if _t(now) < SIP_READY:
            log(f"Box watch: IEX has only {got} of the six 9:00-9:30 five-minute bars; waiting until "
                f"{SIP_READY:%H:%M} to build the box from the SIP feed.")
            return False
        start = pd.Timestamp(datetime.combine(d, time(9, 0), tzinfo=ET))
        sip = rules.minute_frame(fetch_sip_box())
        sip = sip[(sip.index >= start) & (sip.index < start + pd.Timedelta(minutes=30))]
        today = pd.concat([sip, today[today.index >= start + pd.Timedelta(minutes=30)]])
        five = rules.bars(today, 5, now=cutoff)
        bx = rules.box(five)
        if bx is None:
            return decide("no_box", f"the 9:00-9:30 box is incomplete on both feeds (IEX {got} of 6, SIP "
                                    f"{_box_buckets(five)} of 6 five-minute bars; a half day or a data gap)")
        log(f"Box watch: IEX had only {got} of the six box bars, so the box comes from the SIP feed "
            f"(${bx[0]:.2f} high, ${bx[1]:.2f} low).")
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
    if hist15.attrs.get("problem"):
        return decide("skipped", f"{seen}, but {hist15.attrs['problem']}", box=list(bx))
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
    want = math.floor(account.equity * RISK_PCT / sig.risk) if sig.risk > 0 else 0
    if qty == 0:
        if want < 1:
            why = (f"1% of ${account.equity:,.2f} is less than one share's risk of ${sig.risk:.2f}")
        else:
            why = (f"not even one share at ${sig.entry:,.2f} fits in 95% of the ${account.buying_power:,.2f} "
                   "buying power")
        return decide("skipped", f"{seen}, but {why}", box=list(bx), signal=signal)
    risk_pct = qty * sig.risk / account.equity
    if qty < want:
        risking = (f"{risk_pct * 100:.2f}% of ${account.equity:,.2f}, not 1%: {want} shares would cost "
                   f"${want * sig.entry:,.2f}, more than 95% of the ${account.buying_power:,.2f} buying power, "
                   f"so the position is sized down to {qty} shares")
    else:
        risking = f"1% of ${account.equity:,.2f}"

    side = "buy" if sig.side == "long" else "sell"
    what = (f"Box breakout: {seen}. Sending a bracket order to {'buy' if side == 'buy' else 'short'} {qty} "
            f"{SYMBOL} at market, stop ${sig.stop:.2f}, target ${sig.target:.2f} (3R), risking about "
            f"${qty * sig.risk:,.2f} ({risking}). Order id {cid}.")
    if dry_run:
        say("thought", what + " (dry run, nothing sent)")
        return True
    if book.get("halted", False):         # the kill switch or a pause may have landed during this step
        return decide("skipped", f"{seen}, but trading was paused just before the order went out ("
                                 + (book.get("halt_reason") or "switched off") + ")", box=list(bx), signal=signal)
    if book.get("strategy", "selloff") != "box":
        return decide("skipped", f"{seen}, but the strategy changed to {book.get('strategy', 'selloff')} just "
                                 "before the order went out", box=list(bx), signal=signal)
    try:
        broker.submit_bracket_order(SYMBOL, qty, side, sig.stop, sig.target, cid)
    except Exception as err:
        try:
            taken = broker.get_order(cid) is not None
        except Exception:
            # Can't tell whether Alpaca took it. Retry next minute: the id check at the top of this step
            # finds the order if it went through, so a retry never sends a second entry.
            log(f"Box breakout: the bracket order {cid} errored ({type(err).__name__}: {err}) and Alpaca "
                "couldn't be asked whether it arrived; checking again next minute.")
            raise err
        if not taken:
            return decide("failed", f"{seen}, but the bracket order failed: {type(err).__name__}: {err}",
                          kind="fail", box=list(bx), signal=signal)
    book.set("box_today", {"day": day, "status": "sent", "reason": seen, "box": list(bx), "signal": signal,
                           "order": {"id": cid, "qty": qty, "side": side, "stop": sig.stop, "target": sig.target,
                                     "risk": sig.risk, "equity": account.equity,
                                     "risk_pct": round(risk_pct, 6)}})
    say("progress", what)
    return True


def _box_buckets(five: pd.DataFrame) -> int:
    """How many of the six 9:00-9:25 five-minute bars exist."""
    return int(((five.index.hour == 9) & (five.index.minute < 30)).sum())


def watch(broker, book: PaperBook, hist15: pd.DataFrame, fetch_today: Callable[[datetime], pd.DataFrame],
          clock: Callable[[], datetime] = lambda: datetime.now(ET), sleep=_time.sleep, dry_run: bool = False,
          log: Callable[[str], None] = print,
          fetch_sip_box: Callable[[], pd.DataFrame] | None = None) -> dict | None:
    """Once a minute until 10:31: fetch today's minutes and take one step. Stops early once decided.

    A failed fetch or step (a broker hiccup) is logged and retried the next minute; the Alpaca order-id
    check at the top of `step` makes a retry safe against a second entry."""
    problem = None
    while True:
        now = clock()
        decided = False
        try:
            today = fetch_today(now)
            decided = step(broker, book, hist15, today, now, dry_run=dry_run, log=log, sleep=sleep,
                           fetch_sip_box=fetch_sip_box)
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


FLATTEN_ATTEMPTS, FLATTEN_GAP = 7, 30.0   # a failed flatten is retried for about 3 minutes
SETTLE_POLLS, SETTLE_NAP = 20, 0.5          # after cancelling, wait up to ~10 s for every leg to be final
SETTLED = {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}


def _find_today(broker, book: PaperBook, state: dict | None, now: datetime, dry_run: bool,
                log: Callable[[str], None]) -> dict | None:
    """Today's box order at Alpaca when the book doesn't track it (a send whose outcome was never
    recorded). The rebuilt record, or None if Alpaca has no such order or today's trade is already closed."""
    day = now.date().isoformat()
    if state and state.get("day") == day and state.get("status") == "closed":
        return None
    cid = order_id(now.date())
    existing = broker.get_order(cid)
    if existing is None:
        return None
    rec = _recovered(broker, cid, day, existing)
    if state and state.get("day") == day:
        rec = {**{k: state[k] for k in ("box", "signal") if k in state}, **rec}
    rec["reason"] = "found at Alpaca by the flatten job; my records hadn't tracked it"
    text = (f"Box flatten: today's box order {cid} is at Alpaca ({existing.status}) although my records said "
            f"{(state or {}).get('status', 'nothing') if (state or {}).get('day') == day else 'nothing for today'}; "
            "closing it now.")
    log(text)
    if not dry_run:
        book.log(day, "fail", text)
        book.set("box_today", rec)
    return rec


def flatten(broker, book: PaperBook, now: datetime | None = None, dry_run: bool = False, sleep=_time.sleep,
            wait: int = 30, log: Callable[[str], None] = print, attempts: int = FLATTEN_ATTEMPTS,
            gap: float = FLATTEN_GAP) -> dict | None:
    """Cancel Stan's open box orders, close what the box trade still holds, and record its R from the fills.

    Whatever the book says, today's `stan-box-YYYYMMDD` order is looked up at Alpaca, so a trade whose
    send was never recorded is still closed. A failed attempt is retried every `gap` seconds, `attempts`
    times in all (about 3 minutes)."""
    now = (now or datetime.now(ET)).astimezone(ET)
    state = book.get("box_today")
    if not state or state.get("status") not in OPEN or not state.get("order"):
        found = _find_today(broker, book, state, now, dry_run, log)
        if found is None:
            log("Box flatten: no open box trade, nothing to close.")
            return state
        state = found
    day = state["day"]

    def say(kind: str, text: str) -> None:
        log(text)
        if not dry_run:
            book.log(day, kind, text)

    if dry_run:
        _, _, left = _fills(broker, state)
        say("thought", f"Box flatten: would cancel the open {PREFIX} orders and close {left:g} {SYMBOL} "
                       "(dry run, nothing sent).")
        return state

    out = state
    for attempt in range(1, attempts + 1):
        final = attempt == attempts
        try:
            out = _flatten_once(broker, book, state, now, sleep, wait, say, final)
        except Exception as err:
            out = _failed(book, book.get("box_today") or state, say, f"{type(err).__name__}: {err}", final)
        if out.get("status") != "flatten_failed":
            return out
        if not final:
            log(f"Box flatten attempt {attempt} of {attempts} failed ({out.get('reason')}); "
                f"trying again in {gap:g} s.")
            sleep(gap)
            state = book.get("box_today") or out
    return out


def _settle(broker, state: dict, sleep) -> bool:
    """After cancelling: wait until the entry, both legs and any close orders are final (filled or
    cancelled), so a leg filling during the cancel is counted before deciding what's left to close."""
    for i in range(SETTLE_POLLS):
        parent, legs = broker.get_bracket(state["order"]["id"])
        orders = [o for o in [parent, *legs, *(broker.get_order(c) for c in state.get("close_orders", []))]
                  if o is not None]
        if all(o.status in SETTLED for o in orders):
            return True
        if i < SETTLE_POLLS - 1:
            sleep(SETTLE_NAP)
    return False


def _flatten_once(broker, book: PaperBook, state: dict, now: datetime, sleep, wait: int, say, final: bool) -> dict:
    broker.cancel_orders(PREFIX)
    if not _settle(broker, state, sleep):
        return _failed(book, state, say, "the cancelled box orders still weren't final after about 10 seconds",
                       final)
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
                return _failed(book, state, say, f"couldn't send the closing order ({why})", final)
            for _ in range(wait):
                o = broker.get_order(cid)
                if o is not None and o.filled_qty >= left - 1e-9:
                    break
                sleep(1)
            parent, fills, left = _fills(broker, state)
            if left > 1e-9:
                return _failed(book, state, say, f"the closing order {cid} hasn't filled yet ({left:g} shares left)",
                               final)

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


def _failed(book: PaperBook, state: dict, say, why: str, final: bool = True) -> dict:
    """Mark the trade flatten_failed (keeping its close orders). Only the last attempt reports it."""
    state = {**state, "status": "flatten_failed", "reason": why}
    book.set("box_today", state)
    if final:
        say("fail", f"Box flatten problem: {why}. The position is still open with no stop (its stop and target "
                    "were cancelled). The next run tries again; close it by hand if it can't wait.")
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


def previous_session(day: date, sessions: Callable[[date, date], list[date]] | None = None) -> date:
    """The trading session before `day`: from `sessions(start, end)` (Alpaca's calendar) when given and
    working, else the previous weekday."""
    if sessions is not None:
        try:
            earlier = [x for x in sessions(day - timedelta(days=10), day - timedelta(days=1)) if x < day]
            if earlier:
                return max(earlier)
        except Exception:
            pass
    d = day - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def trend_history(client, db, start: datetime, backfill=None, load=None, log: Callable[[str], None] = print,
                  prev_session: date | None = None):
    """Finished 15-minute SIP bars before `start` (9:00 today), after a quick top-up of the stored bars.

    The top-up only runs when the stored bars are at most 30 days behind, so a missing or stale database
    can't hold up the watch. Without recent bars the history is empty and the watch skips the day.
    If the stored bars still end before `prev_session` after the top-up (it failed), the EMAs would be
    stale: the frame comes back with `attrs["problem"]` set and the watch skips the day with that reason."""
    from agent.stan.box import data
    backfill, load = backfill or data.backfill, load or data.load
    empty = pd.DataFrame(columns=["o", "h", "l", "c"], index=pd.DatetimeIndex([], tz=ET), dtype=float)
    last = _last_stored(db)
    if last is None or start - last > timedelta(days=TOP_UP_DAYS):
        log(f"The SIP bars in {db} " + (f"end {last:%Y-%m-%d}" if last else "are missing")
            + f", more than {TOP_UP_DAYS} days behind; not fetching them during the watch. "
            "Run python -m agent.stan.box.data backfill first. No trend history today.")
        return empty
    failed = None
    try:
        n = backfill(client, db, log=log)
        log(f"Topped up {db} with {n:,} SIP bars.")
    except Exception as err:
        failed = f"{type(err).__name__}: {err}"
        log(f"Couldn't top up the SIP bars ({failed}); using what's stored.")
    try:
        stored = load(db)
    except Exception as err:
        log(f"Couldn't read {db} ({type(err).__name__}: {err}); no trend history.")
        return empty
    out = rules.bars(stored[stored.index < pd.Timestamp(start)], 15)
    last = _last_stored(db)
    if prev_session is not None and last is not None and last.date() < prev_session:
        out.attrs["problem"] = (f"the stored SIP bars end {last:%Y-%m-%d}, before the last session "
                                f"({prev_session:%Y-%m-%d}), and the top-up "
                                + (f"failed ({failed})" if failed else "brought nothing newer")
                                + ", so the 15-minute trend would be stale")
        log("Trend history is stale: " + out.attrs["problem"] + ".")
    return out


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
        def sessions(a: date, b: date) -> list[date]:
            from alpaca.trading.requests import GetCalendarRequest
            return [c.date for c in broker._trading.get_calendar(GetCalendarRequest(start=a, end=b))]

        hist15 = trend_history(client, db, start, prev_session=previous_session(start.date(), sessions))
        last = hist15.index.max() if len(hist15) else None
        print(f"Trend history: {len(hist15):,} 15-minute bars from the SIP feed"
              + (f", last one {last:%Y-%m-%d %H:%M}." if last is not None else "."))
        print("Today's minutes from 9:00 come from the IEX feed (SIP is about 15 minutes behind).")

        def fetch_today(now: datetime) -> pd.DataFrame:
            return rules.minute_frame(fetch_alpaca(client, SYMBOL, start, now, feed="iex"))

        sip_box: list[pd.DataFrame] = []

        def fetch_sip_box() -> pd.DataFrame:      # 9:00-9:30 from SIP, fetched once it is complete
            if not sip_box:
                got = rules.minute_frame(fetch_alpaca(client, SYMBOL, start, start + timedelta(minutes=30),
                                                      feed="sip"))
                if _box_buckets(rules.bars(got, 5)) == 6:
                    sip_box.append(got)
                return got
            return sip_box[0]

        state = watch(broker, book, hist15, fetch_today, dry_run=args.dry_run, fetch_sip_box=fetch_sip_box)
        print(f"Box watch done: {(state or {}).get('status', 'nothing recorded')}"
              + (" (dry run, nothing sent or saved)" if args.dry_run else "") + ".")
        return 0
    finally:
        book.close()


if __name__ == "__main__":
    sys.exit(main())
