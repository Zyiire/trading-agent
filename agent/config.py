"""Settings, loaded from environment variables (a git-ignored .env file).

Keys are never hard-coded and never printed. This phase is paper-only:
live trading stays locked until the go-live checks in the spec pass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time so tests run without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

DEFAULT_WATCHLIST = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "IWM")


class ConfigError(RuntimeError):
    """Raised when settings are missing or unsafe."""


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    paper: bool = True
    db_path: Path = Path("data/agent.db")
    watchlist: tuple[str, ...] = DEFAULT_WATCHLIST
    # Risk settings from the spec (used by the risk engine in the next phase).
    starting_capital: float = 50.0
    kill_switch_drawdown: float = 0.20
    # Options chain snapshots: how far out and how wide around the stock price.
    snapshot_interval_minutes: int = 15
    snapshot_max_days_to_expiry: int = 60
    snapshot_strike_band: float = 0.15

    @property
    def kill_switch_equity(self) -> float:
        return round(self.starting_capital * (1 - self.kill_switch_drawdown), 2)


def load_settings(env_file: str | os.PathLike | None = ".env") -> Settings:
    if load_dotenv is not None and env_file and Path(env_file).exists():
        load_dotenv(env_file)

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not secret_key:
        raise ConfigError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY. Copy .env.example to .env "
            "and paste in your Alpaca PAPER keys."
        )

    paper = _as_bool(os.getenv("ALPACA_PAPER"), default=True)
    if not paper:
        raise ConfigError(
            "Live trading is locked in this phase. Set ALPACA_PAPER=true. "
            "Real money is only enabled after the go-live checks pass."
        )

    raw_watchlist = os.getenv("WATCHLIST", "")
    watchlist = tuple(s.strip().upper() for s in raw_watchlist.split(",") if s.strip())

    return Settings(
        api_key=api_key,
        secret_key=secret_key,
        paper=paper,
        db_path=Path(os.getenv("DB_PATH", "data/agent.db")),
        watchlist=watchlist or DEFAULT_WATCHLIST,
        starting_capital=float(os.getenv("STARTING_CAPITAL", "50")),
        kill_switch_drawdown=float(os.getenv("KILL_SWITCH_DRAWDOWN", "0.20")),
        snapshot_interval_minutes=int(os.getenv("SNAPSHOT_INTERVAL_MINUTES", "15")),
        snapshot_max_days_to_expiry=int(os.getenv("SNAPSHOT_MAX_DAYS_TO_EXPIRY", "60")),
        snapshot_strike_band=float(os.getenv("SNAPSHOT_STRIKE_BAND", "0.15")),
    )
