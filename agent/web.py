"""Stan's desktop web app: one page to watch the paper account live, pick a risk tier and run his trade.

    python -m agent.web          # opens http://127.0.0.1:8000 in your browser
    python -m agent.web --demo   # made-up account and prices, to try the page with no Alpaca keys

It serves the page in agent/webapp/ and adds a few endpoints to the phone API:
    GET  /live                 account, positions and market clock, straight from Alpaca
    GET  /live/chart?symbol=   5-minute prices for one holding
    POST /paper/use            choose box, selloff, low, medium or high
    POST /paper/run            preview (dry run) or send Stan's trade; GET /paper/job reports on it
    POST /paper/pause, /paper/resume

Paper account only: the broker adapter refuses live settings. The endpoints that change anything
answer only requests from this computer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time as _time
import webbrowser
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.api import create_app
from agent.broker.base import AccountInfo, MarketClock, OrderInfo, PositionInfo
from agent.stan import tiers
from agent.stan.paper import PaperBook, run_evening, set_strategy
from agent.stan.shadow import ET

WEBAPP = Path(__file__).parent / "webapp"
TIER_SYMBOLS = ["SPY", "QQQ", "TQQQ"]


class UseBody(BaseModel):
    strategy: str


class RunBody(BaseModel):
    dry_run: bool = True


# ---------------------------------------------------------------- demo broker (made-up prices)
class DemoBroker:
    """A pretend paper account whose prices wander a little every few seconds."""

    def __init__(self, prices: dict[str, float], cash: float = 10_000.0, seed: int = 7) -> None:
        self.prices = dict(prices)
        self.cash, self.start_cash = cash, cash
        self.positions: dict[str, list[float]] = {}   # symbol -> [qty, average entry]
        self.orders: dict[str, OrderInfo] = {}
        self._rng = np.random.default_rng(seed)
        self._last = _time.time()
        self.paths = {s: deque(self._seed_path(p), maxlen=400) for s, p in self.prices.items()}
        self.last_equity = cash

    def _seed_path(self, price: float) -> list[tuple[float, float]]:
        steps = self._rng.normal(0, 0.0011, 156)
        closes = price * np.exp(np.cumsum(steps) - np.sum(steps))
        now = _time.time()
        return [(now - 300 * (156 - k), float(c)) for k, c in enumerate(closes, 1)]

    def _walk(self) -> None:
        now = _time.time()
        if now - self._last < 2:
            return
        self._last = now
        market = self._rng.normal(0.00003, 0.0006)
        for s in self.prices:
            beta = 3.0 if s == "TQQQ" else 1.0
            self.prices[s] *= 1 + beta * market + self._rng.normal(0, 0.0002)
            self.paths[s].append((now, self.prices[s]))

    def get_account(self) -> AccountInfo:
        self._walk()
        equity = self.cash + sum(q * self.prices[s] for s, (q, _) in self.positions.items())
        return AccountInfo("demo", "ACTIVE", True, equity, self.cash, self.cash, None, 2, False, False, "1",
                           self.last_equity)

    def get_positions(self) -> list[PositionInfo]:
        self._walk()
        return [PositionInfo(s, "us_equity", q, avg, q * self.prices[s], q * (self.prices[s] - avg))
                for s, (q, avg) in self.positions.items() if q > 1e-9]

    def get_clock(self) -> MarketClock:
        now = datetime.now(timezone.utc)
        return MarketClock(now, True, now + timedelta(hours=18), now + timedelta(hours=3))

    def _fill(self, symbol: str, qty: float, side: str, oid: str) -> OrderInfo:
        if oid in self.orders:
            raise RuntimeError("client_order_id must be unique")
        px = self.prices.setdefault(symbol, 100.0)
        self.paths.setdefault(symbol, deque(self._seed_path(px), maxlen=400))
        q, avg = self.positions.get(symbol, [0.0, 0.0])
        if side == "buy":
            avg = (q * avg + qty * px) / (q + qty)
            q += qty
            self.cash -= qty * px
        else:
            qty = min(qty, q)
            q -= qty
            self.cash += qty * px
        self.positions[symbol] = [q, avg]
        self.orders[oid] = OrderInfo(oid, symbol, side, qty, "filled", qty, px, datetime.now(timezone.utc))
        return self.orders[oid]

    def submit_fractional_order(self, symbol, qty, side, client_order_id):
        return self._fill(symbol, float(qty), side, client_order_id)

    def submit_open_order(self, symbol, qty, side, client_order_id):
        return self._fill(symbol, float(qty), side, client_order_id)

    def get_order(self, client_order_id):
        return self.orders.get(client_order_id)

    def cancel_open_orders(self) -> int:
        return 0


def demo_setup(journal_path: str) -> tuple[DemoBroker, Callable[[list[str]], dict]]:
    """A demo broker, a history loader that agrees with its prices, and a journal with some past."""
    from agent.stan.train import demo_history

    hist = demo_history(TIER_SYMBOLS, years=2, seed=21)
    qqq = hist["QQQ"]["close"].pct_change().fillna(0.0)
    t = 60 * (1 + 3 * qqq).cumprod()
    hist["TQQQ"] = pd.DataFrame({"open": t.shift(1).fillna(t.iloc[0]), "high": t * 1.01, "low": t * 0.99,
                                 "close": t, "volume": 1e6})
    broker = DemoBroker({s: float(df["close"].iloc[-1]) for s, df in hist.items()})
    book = PaperBook(journal_path)
    try:
        if book.get("start_equity") is None:
            book.set("strategy", "medium")
            evening = datetime.combine(datetime.now(ET).date(), datetime.min.time(), tzinfo=ET).replace(hour=17)
            run_evening(broker, book, hist, now=evening)
            rng = np.random.default_rng(3)
            days = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.offsets.BDay(1), periods=34)
            eq = 10_000 * np.cumprod(1 + rng.normal(0.0009, 0.006, len(days)))
            eq = eq * (broker.get_account().equity / eq[-1])
            for d, e in zip(days, eq):
                book.record_equity(str(d.date()), float(e), float(e) * 0.3)
            book.set("start_equity", float(eq[0]))
            book.set("started", str(days[0].date()))
            broker.last_equity = float(eq[-1]) * 0.996
        _seed_demo_arena(journal_path)
    finally:
        book.close()
    return broker, lambda symbols: hist


def _seed_demo_arena(journal_path: str) -> None:
    """Copy the practice arena results into the demo journal so the tier cards have numbers."""
    from agent.stan.journal import Journal

    demo_json = Path(__file__).parents[1] / "stan-app" / "src" / "demo.json"
    j = Journal(journal_path)
    try:
        if j.meta().get("arena") is None and demo_json.exists():
            arena = json.loads(demo_json.read_text())["/stan"].get("arena")
            if arena:
                j.set_meta("arena", arena)
                j.set_meta("mode", "demo")
    finally:
        j.close()


# ---------------------------------------------------------------- the app
def create_web_app(journal_path: str | Path, broker, history_loader: Callable[[list[str]], dict],
                   demo: bool = False, token: str | None = None,
                   chart_loader: Callable[[str], list[dict]] | None = None,
                   train_starter: Callable[[], object] | None = None) -> FastAPI:
    app = create_app(journal_path, token)
    journal_path = str(journal_path)
    ticks: deque = deque(maxlen=3000)
    cache = {"at": 0.0, "live": None}
    job = {"running": False, "dry_run": True, "started": None, "finished": None, "actions": [], "error": None,
           "strategy": None, "halted": False, "reason": None}
    lock = threading.Lock()

    def local_only(request: Request) -> None:
        host = request.client.host if request.client else ""
        origin = request.headers.get("origin", "")
        ok_origin = not origin or origin.split("//")[-1].split(":")[0] in ("127.0.0.1", "localhost", "[::1]", "testserver")
        if host not in ("127.0.0.1", "::1", "localhost", "testclient") or not ok_origin:
            raise HTTPException(status_code=403, detail="Only this computer can change Stan's trading.")

    def book() -> PaperBook:
        return PaperBook(journal_path)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEBAPP / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/assets/{name}", include_in_schema=False)
    def asset(name: str) -> FileResponse:
        path = WEBAPP / name
        if name not in ("stan.png",) or not path.exists():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(path, headers={"Cache-Control": "max-age=3600"})

    training = {"proc": None}

    def _start_training():
        cmd = [sys.executable, "-m", "agent.stan.train", "--journal", journal_path]
        if demo:
            cmd += ["--demo", "--pace", "0.4"]
        return subprocess.Popen(cmd, cwd=str(Path(__file__).parents[1]), stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

    def _training_running() -> bool:
        proc = training["proc"]
        return proc is not None and proc.poll() is None

    @app.get("/train/status")
    def train_status() -> dict:
        return {"running": _training_running()}

    @app.post("/train/start", dependencies=[Depends(local_only)])
    def train_start() -> dict:
        """Start a new school term in the background. It rewrites Stan's study notes, not his trades."""
        if _training_running():
            raise HTTPException(status_code=409, detail="Stan is already in class.")
        training["proc"] = (train_starter or _start_training)()
        return {"ok": True}

    def _is_local(request: Request) -> bool:
        return (request.client.host if request.client else "") in ("127.0.0.1", "::1", "localhost", "testclient")

    @app.get("/live")
    def live(request: Request) -> dict:
        """Account, positions and clock. `can_control` tells the page whether this device may change anything."""
        return {**_live(), "can_control": _is_local(request)}

    def _live() -> dict:
        now = _time.time()
        if cache["live"] is not None and now - cache["at"] < 3:
            return cache["live"]
        try:
            acct = broker.get_account()
            positions = broker.get_positions()
            clock = broker.get_clock()
        except Exception as err:
            raise HTTPException(status_code=502, detail=f"Couldn't reach Alpaca: {type(err).__name__}: {err}")
        b = book()
        try:
            start = b.get("start_equity")
            strategy = b.get("strategy", "selloff")
            halted, reason = bool(b.get("halted", False)), b.get("halt_reason")
            plan_ = b.get("tier_plan")
            low_fraction = b.get("low_fraction", tiers.DEFAULT_LOW_FRACTION)
            last_day = b.get("last_day")
        finally:
            b.close()
        rows = []
        for p in positions:
            price = (p.market_value / p.qty) if p.market_value and p.qty else None
            rows.append({"symbol": p.symbol, "qty": p.qty, "avg_entry": p.avg_entry_price, "price": price,
                         "value": p.market_value, "pl": p.unrealized_pl,
                         "pl_pct": (price / p.avg_entry_price - 1) if price and p.avg_entry_price else None})
        rows.sort(key=lambda r: -(r["value"] or 0))
        ticks.append({"t": int(now * 1000), "equity": acct.equity,
                      "prices": {r["symbol"]: r["price"] for r in rows if r["price"]}})
        out = {
            "demo": demo, "paper": acct.is_paper, "equity": acct.equity, "cash": acct.cash,
            "last_equity": acct.last_equity,
            "day_change": (acct.equity / acct.last_equity - 1) if acct.last_equity else None,
            "start_equity": start, "since_start": (acct.equity / start - 1) if start else None,
            "kill_floor": start * 0.8 if start else None,
            "strategy": strategy, "halted": halted, "halt_reason": reason, "tier_plan": plan_,
            "low_fraction": low_fraction, "last_run_day": last_day,
            "market": {"is_open": clock.is_open, "next_open": clock.next_open.isoformat(),
                       "next_close": clock.next_close.isoformat()},
            "positions": rows, "ticks": list(ticks)[-720:], "now": int(now * 1000),
        }
        cache.update(at=now, live=out)
        return out

    @app.get("/live/chart")
    def live_chart(symbol: str) -> dict:
        symbol = symbol.upper()[:8]
        if not symbol.replace(".", "").isalnum():
            raise HTTPException(status_code=400, detail="Bad symbol")
        if isinstance(broker, DemoBroker):
            pts = [{"t": int(t * 1000), "c": c} for t, c in broker.paths.get(symbol, [])]
        else:
            pts = (chart_loader or yahoo_5m)(symbol)
        return {"symbol": symbol, "points": pts}

    @app.post("/paper/use", dependencies=[Depends(local_only)])
    def use(body: UseBody) -> dict:
        if job["running"]:
            raise HTTPException(status_code=409, detail="Stan is in the middle of a run. Try again in a moment.")
        b = book()
        try:
            msg = set_strategy(b, journal_path, body.strategy)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        finally:
            b.close()
        cache["live"] = None
        return {"ok": True, "message": msg, "strategy": body.strategy}

    def _run(dry_run: bool) -> None:
        b = book()
        try:
            strategy = b.get("strategy", "selloff")
            job["strategy"] = strategy
            open_syms = [t["symbol"] for t in b.trades(["pending_entry", "open", "pending_exit"])]
            if strategy == "selloff":
                from agent.stan.train import DEFAULT_SYMBOLS
                symbols = sorted(set(DEFAULT_SYMBOLS) | set(TIER_SYMBOLS))
            else:
                symbols = sorted(set(TIER_SYMBOLS) | set(open_syms))
            history = history_loader(symbols)
            if "SPY" not in history:
                raise RuntimeError("Couldn't load prices, so nothing was done.")
            result = run_evening(broker, b, history, dry_run=dry_run)
            job.update(actions=result.get("actions") or ["Nothing to do right now."], halted=result.get("halted", False),
                       reason=result.get("reason"), day=result.get("day"))
        except Exception as err:
            job.update(error=f"{type(err).__name__}: {err}")
        finally:
            b.close()
            job.update(running=False, finished=datetime.now(timezone.utc).isoformat())
            cache["live"] = None

    @app.post("/paper/run", dependencies=[Depends(local_only)])
    def run(body: RunBody) -> dict:
        with lock:
            if job["running"]:
                raise HTTPException(status_code=409, detail="Stan is already running.")
            job.update(running=True, dry_run=body.dry_run, started=datetime.now(timezone.utc).isoformat(),
                       finished=None, actions=[], error=None, halted=False, reason=None)
        threading.Thread(target=_run, args=(body.dry_run,), daemon=True).start()
        return {"ok": True}

    @app.get("/paper/job")
    def job_status() -> dict:
        return dict(job)

    @app.post("/paper/pause", dependencies=[Depends(local_only)])
    def pause() -> dict:
        b = book()
        try:
            b.set("halted", True)
            b.set("halt_reason", "Paused by Zy")
            b.log(datetime.now(ET).date().isoformat(), "milestone", "Zy paused paper trading.")
        finally:
            b.close()
        cache["live"] = None
        return {"ok": True}

    @app.post("/paper/resume", dependencies=[Depends(local_only)])
    def resume() -> dict:
        b = book()
        try:
            b.set("halted", False)
            b.set("halt_reason", None)
            b.log(datetime.now(ET).date().isoformat(), "milestone", "Zy restarted paper trading.")
        finally:
            b.close()
        cache["live"] = None
        return {"ok": True}

    return app


_chart_cache: dict[str, tuple[float, list[dict]]] = {}


def yahoo_5m(symbol: str) -> list[dict]:
    """The last two sessions of 5-minute closes from Yahoo (cached for a minute)."""
    hit = _chart_cache.get(symbol)
    if hit and _time.time() - hit[0] < 60:
        return hit[1]
    pts: list[dict] = []
    try:
        import yfinance as yf

        df = yf.download(symbol, period="5d", interval="5m", auto_adjust=True, progress=False)
        if df is not None and len(df):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            days = sorted(set(close.index.date))[-2:]
            close = close[[d in days for d in close.index.date]]
            pts = [{"t": int(ts.timestamp() * 1000), "c": float(c)} for ts, c in close.items()]
    except Exception:
        pts = []
    _chart_cache[symbol] = (_time.time(), pts)
    return pts


def _lan_ip() -> str:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))   # no packet is sent; this just picks the outgoing interface
            return sock.getsockname()[0]
    except OSError:
        return "this-computer's-IP"


def main(argv=None) -> int:
    import uvicorn

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Stan's desktop web app.")
    parser.add_argument("--demo", action="store_true", help="made-up account and prices; no Alpaca keys needed")
    parser.add_argument("--port", type=int, default=int(os.getenv("STAN_API_PORT", "8000")))
    parser.add_argument("--host", default=os.getenv("STAN_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--phone", action="store_true",
                        help="also let phones on your Wi-Fi view the page (view only; changes stay on this PC)")
    parser.add_argument("--journal", default=None)
    parser.add_argument("--db", default="data/agent.db")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.demo:
        journal = args.journal or "data/stan-demo.db"
        broker, loader = demo_setup(journal)
    else:
        from agent.broker.alpaca import AlpacaBroker
        from agent.config import ConfigError, load_settings
        from agent.stan.train import real_history

        journal = args.journal or os.getenv("STAN_JOURNAL", "data/stan.db")
        try:
            broker = AlpacaBroker(load_settings())
        except ConfigError as err:
            print(f"Setup problem: {err}\nTo look around without Alpaca keys, run: python -m agent.web --demo")
            return 1

        def loader(symbols: list[str]) -> dict:
            return real_history(symbols, 2, args.db)[0]

    app = create_web_app(journal, broker, loader, demo=args.demo, token=os.getenv("STAN_API_TOKEN") or None)
    if args.phone:
        args.host = "0.0.0.0"
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}"
    print(f"Stan is at {url}" + ("  (demo: made-up account and prices)" if args.demo else ""))
    if args.phone:
        print(f"On your phone (same Wi-Fi), open http://{_lan_ip()}:{args.port}  (view only)")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
