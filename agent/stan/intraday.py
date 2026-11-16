"""Level play, live: Stan trades QQQ during the session on the paper account.

Every time a 5-minute bar closes, Stan:
  1. rebuilds today's levels from the previous 3 sessions' volume profile (once a day),
  2. if he holds a position, checks the bar against his stop, target and break-even, and sells at
     market when one is hit, or at 3:45 PM at the latest,
  3. if he's flat (and trading is on, the kill switch hasn't tripped, and he's made fewer than 3 trades
     today), checks for a level lost or reclaimed with a fair value gap, and enters at market.

Longs buy QQQ shares. "Losing a level" (the reels' put side) sells QQQ short, which needs a margin
account; on a cash account Stan skips shorts and says so. Size: 1% of the account at risk per trade,
never more than the account's value in shares. Paper account only.

The state of the open position lives in the paper book, so a restart picks up where he left off, and
a heartbeat stops a second copy (the web app and the scheduled run) from trading at the same time.
"""
from __future__ import annotations

import math
import time as _time
from datetime import datetime, time, timedelta
from typing import Callable

import pandas as pd

from agent.broker.base import Broker
from agent.stan import levels as L
from agent.stan.paper import PaperBook
from agent.stan.price_action.series import from_sessions
from agent.stan.shadow import ET

SYMBOL = "QQQ"
RISK_PCT = 0.01
HEARTBEAT_S = 120
SESSION_OPEN, SESSION_CLOSE = time(9, 30), time(16, 0)


def sessions_from_frame(df: pd.DataFrame, now: datetime) -> list[tuple[str, list[dict]]]:
    """Regular-session 5-minute bars grouped by day, keeping only bars that have closed by `now`."""
    if df is None or not len(df):
        return []
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    et = idx.tz_convert(ET)
    out: dict[str, list[dict]] = {}
    for ts, row in zip(et, df.itertuples(index=False)):
        t = ts.time()
        if not (SESSION_OPEN <= t < SESSION_CLOSE) or ts + timedelta(minutes=5) > now:
            continue
        slot = (ts.hour * 60 + ts.minute - 570) // 5
        out.setdefault(str(ts.date()), []).append({"slot": int(slot), "o": float(row.open), "h": float(row.high),
                                                   "l": float(row.low), "c": float(row.close),
                                                   "v": float(row.volume or 0)})
    return sorted(out.items())


def yahoo_bars(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, period="7d", interval="5m", auto_adjust=True, progress=False, prepost=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()


class LiveLevels:
    def __init__(self, broker: Broker, book: PaperBook, bars: Callable[[str], pd.DataFrame] = yahoo_bars,
                 say: Callable[[str, str], None] | None = None, cfg: L.LevelConfig = L.LevelConfig(),
                 dry_run: bool = False) -> None:
        self.broker, self.book, self.bars, self.cfg, self.dry_run = broker, book, bars, cfg, dry_run
        self.actions: list[str] = []
        self._say = say

    # ---------------- helpers
    def say(self, kind: str, text: str, day: str) -> None:
        self.actions.append(text)
        if self._say:
            self._say(kind, text)
        if not self.dry_run:
            self.book.log(day, kind, text)

    def other_runner_alive(self, now: datetime, me: str) -> bool:
        hb = self.book.get("levels_heartbeat")
        return bool(hb and hb.get("who") != me and now.timestamp() - hb.get("at", 0) < HEARTBEAT_S)

    def beat(self, now: datetime, me: str) -> None:
        if not self.dry_run:
            self.book.set("levels_heartbeat", {"who": me, "at": now.timestamp()})

    def _order(self, side: str, qty: int, oid: str, day: str) -> bool:
        if self.dry_run:
            return True
        try:
            self.broker.submit_fractional_order(SYMBOL, qty, side, oid)
            return True
        except Exception as err:
            try:
                if self.broker.get_order(oid) is not None:
                    return True
            except Exception:
                pass
            self.say("fail", f"Couldn't send the QQQ {side} order: {type(err).__name__}: {err}", day)
            return False

    # ---------------- one step, after each 5-minute bar
    def step(self, now: datetime, me: str = "runner") -> dict:
        """Look at the latest closed bar and act. Returns a short status for the web app."""
        self.actions = []
        sessions = sessions_from_frame(self.bars(SYMBOL), now)
        if not sessions or sessions[-1][0] != str(now.date()):
            return {"status": "waiting", "detail": "No bars from today yet."}
        day = sessions[-1][0]
        s = from_sessions(SYMBOL, sessions)
        lv = L.session_levels(s, self.cfg)[len(s.days) - 1]
        i = len(s) - 1
        slot = int(s.slot[i])
        t = self.cfg.trade
        if self.book.get("levels_day") != day:
            names = ", ".join(f"{n} {p:.2f}" for n, p in sorted(lv, key=lambda x: -x[1])) or "none (need 3 sessions)"
            self.say("thought", f"Level play: today's QQQ levels from the last 3 sessions' volume profile: {names}.", day)
            if not self.dry_run:
                self.book.set("levels_day", day)
                self.book.set("levels_trades", 0)
                self.book.set("levels_today", [[n, round(p, 4)] for n, p in lv])
        pos = self.book.get("levels_pos")
        bar_key = f"{day}-{slot}"
        if self.book.get("levels_last_bar") == bar_key and not self.dry_run:
            return {"status": "holding" if pos else "watching", "position": pos, "levels": lv, "actions": []}

        # manage an open position
        if pos:
            side = pos["side"]
            hi, lo, close = float(s.h[i]), float(s.l[i]), float(s.c[i])
            why = None
            if (side > 0 and lo <= pos["stop"]) or (side < 0 and hi >= pos["stop"]):
                why = "break-even stop" if pos.get("be") else "stop"
            elif (side > 0 and hi >= pos["target"]) or (side < 0 and lo <= pos["target"]):
                why = "target"
            elif slot >= t.flatten_slot:
                why = "3:45 PM, flat for the day"
            if why:
                oid = f"stan-lv-{day.replace('-', '')}-{slot}-exit"
                if self._order("sell" if side > 0 else "buy", pos["qty"], oid, day):
                    r = side * (close - pos["entry"]) / pos["risk"]
                    self.say("win" if r > 0 else "fail",
                             f"Out of QQQ ({why}) at about {close:.2f}: roughly {r:+.1f}R. "
                             f"{'Bought back' if side < 0 else 'Sold'} {pos['qty']} shares.", day)
                    if not self.dry_run:
                        self.book.set("levels_pos", None)
                    pos = None
            elif not pos.get("be") and ((side > 0 and hi >= pos["entry"] + pos["risk"])
                                        or (side < 0 and lo <= pos["entry"] - pos["risk"])):
                pos = {**pos, "stop": pos["entry"], "be": True}
                self.say("progress", "QQQ is 1R in my favour, so my stop moves to break-even.", day)
                if not self.dry_run:
                    self.book.set("levels_pos", pos)

        # look for a new trade
        elif self.book.get("halted", False):
            pass
        elif (self.book.get("levels_trades") or 0) >= t.max_trades_per_day:
            pass
        else:
            sig = L.signal_at(s, i, lv, self.cfg)
            if sig:
                acct = self.broker.get_account()
                entry = float(s.c[i])
                stop = sig.structure - t.stop_buffer if sig.side > 0 else sig.structure + t.stop_buffer
                risk = sig.side * (entry - stop)
                rr = entry + sig.side * t.rr_target * risk
                target = sig.target if sig.target is not None and sig.side * (sig.target - rr) < 0 else rr
                room = sig.side * (target - entry)
                what = f"{'reclaimed' if sig.side > 0 else 'lost'} the {sig.level_name} at {sig.level:.2f}"
                if risk <= 0 or room < t.min_room_r * risk:
                    self.say("thought", f"QQQ {what}, but the next level is under 1R away, so I pass.", day)
                elif sig.side < 0 and (acct.multiplier or "1") == "1":
                    self.say("thought", f"QQQ {what} with a fair value gap: the put side. This is a cash account, "
                                        "so I can't sell short. Passing.", day)
                else:
                    qty = min(math.floor(acct.equity * RISK_PCT / risk), math.floor(acct.equity * 0.95 / entry))
                    if qty < 1:
                        self.say("thought", f"QQQ {what}, but one share is more than 1% risk on this account.", day)
                    else:
                        oid = f"stan-lv-{day.replace('-', '')}-{slot}-{'buy' if sig.side > 0 else 'sell'}"
                        if self._order("buy" if sig.side > 0 else "sell", qty, oid, day):
                            pos = {"side": sig.side, "qty": qty, "entry": entry, "stop": round(stop, 4),
                                   "target": round(target, 4), "risk": round(risk, 4), "be": False, "day": day,
                                   "level": sig.level_name}
                            self.say("progress", f"QQQ {what} with a fair value gap. "
                                                 f"{'Bought' if sig.side > 0 else 'Sold short'} {qty} shares near "
                                                 f"{entry:.2f}; stop {stop:.2f}, target {target:.2f}.", day)
                            if not self.dry_run:
                                self.book.set("levels_pos", pos)
                                self.book.set("levels_trades", (self.book.get("levels_trades") or 0) + 1)
        if not self.dry_run:
            self.book.set("levels_last_bar", bar_key)
        return {"status": "holding" if pos else "watching", "position": pos, "levels": lv, "actions": self.actions}

    # ---------------- a preview for the web app
    def describe(self, now: datetime) -> list[str]:
        """What Stan sees right now, without trading (for Preview)."""
        sessions = sessions_from_frame(self.bars(SYMBOL), now)
        if not sessions:
            return ["Level play: I couldn't load QQQ's 5-minute bars."]
        s = from_sessions(SYMBOL, sessions)
        today = sessions[-1][0] == str(now.date())
        k = len(s.days) - 1 if today else len(s.days)
        lv = L.session_levels(s, self.cfg).get(k) if today else \
            L.profile_levels(*[a[s.day_idx >= len(s.days) - self.cfg.profile_sessions] for a in (s.h, s.l, s.v)], self.cfg)
        names = ", ".join(f"{n} {p:.2f}" for n, p in sorted(lv or [], key=lambda x: -x[1])) or "none yet"
        out = [f"Level play plan: QQQ levels {'today' if today else 'for the next session'}: {names}."]
        pos = self.book.get("levels_pos")
        if pos:
            out.append(f"Holding {pos['qty']} QQQ {'long' if pos['side'] > 0 else 'short'} from {pos['entry']:.2f}; "
                       f"stop {pos['stop']:.2f}, target {pos['target']:.2f}.")
        else:
            out.append(f"Flat. Last price {s.c[-1]:.2f}. I trade when QQQ closes a 5-minute bar through one of these "
                       "levels with a fair value gap.")
        hb = self.book.get("levels_heartbeat")
        live = bool(hb and now.timestamp() - hb.get("at", 0) < HEARTBEAT_S)
        out.append("I'm watching the market now." if live else
                   "I'm not watching yet. Press Start, or the 9:35 AM scheduled run starts me.")
        return out


def run_session(broker: Broker, book: PaperBook, bars: Callable[[str], pd.DataFrame] = yahoo_bars,
                now_fn: Callable[[], datetime] = lambda: datetime.now(ET), sleep: Callable[[float], None] = _time.sleep,
                me: str = "scheduled", say: Callable[[str, str], None] | None = None,
                stop_flag: Callable[[], bool] = lambda: False) -> None:
    """Run until 3:50 PM ET: a step shortly after every 5-minute bar closes."""
    runner = LiveLevels(broker, book, bars, say=say)
    end = time(15, 50)
    while not stop_flag():
        now = now_fn()
        if now.time() >= end or now.weekday() >= 5:
            break
        if runner.other_runner_alive(now, me):
            sleep(30)
            continue
        runner.beat(now, me)
        if now.time() >= time(9, 35):
            try:
                runner.step(now, me)
            except Exception as err:   # a bad data pull shouldn't end the day
                runner.say("fail", f"Level play hit a problem and will retry: {type(err).__name__}: {err}",
                           str(now.date()))
        wait = 300 - (now.minute % 5) * 60 - now.second + 20   # 20 s after the next bar closes
        sleep(max(15, min(wait, 300)))
    acct = broker.get_account()
    book.record_equity(str(now_fn().date()), acct.equity, acct.cash)
    book.set("levels_heartbeat", None)
