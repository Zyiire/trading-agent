"""Stan's journal: tasks, thoughts, exam results and lessons, stored in SQLite.

The trainer writes here; the API (and the phone app) read from here.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', progress REAL NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '', started_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS thoughts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,
    text TEXT NOT NULL, data TEXT
);
CREATE TABLE IF NOT EXISTS exams (
    year INTEGER PRIMARY KEY, n_calls INTEGER, accuracy REAL, baseline_up REAL,
    trades INTEGER, win_rate REAL, profit_factor REAL, stan_return REAL,
    hold_return REAL, passed INTEGER, beat_hold INTEGER
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,
    title TEXT NOT NULL, text TEXT NOT NULL, evidence TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

THOUGHT_KINDS = {"thought", "progress", "win", "fail", "lesson", "milestone"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")  # API can read while the trainer writes
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def reset(self) -> None:
        for table in ("tasks", "thoughts", "exams", "lessons", "meta"):
            self._db.execute(f"DELETE FROM {table}")
        self._db.commit()

    # ----- writing -----
    def set_meta(self, key: str, value: Any) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))
        self._db.commit()

    def plan_tasks(self, tasks: list[tuple[str, str]]) -> None:
        for i, (key, title) in enumerate(tasks, start=1):
            self._db.execute(
                "INSERT OR REPLACE INTO tasks (id, key, title) VALUES (?, ?, ?)", (i, key, title)
            )
        self._db.commit()

    def start_task(self, key: str, detail: str = "") -> None:
        self._db.execute(
            "UPDATE tasks SET status='active', progress=0, detail=?, started_at=? WHERE key=?",
            (detail, now_iso(), key),
        )
        self._db.commit()

    def update_task(self, key: str, progress: float, detail: str | None = None) -> None:
        progress = max(0.0, min(1.0, float(progress)))
        if detail is None:
            self._db.execute("UPDATE tasks SET progress=? WHERE key=?", (progress, key))
        else:
            self._db.execute("UPDATE tasks SET progress=?, detail=? WHERE key=?", (progress, detail, key))
        self._db.commit()

    def finish_task(self, key: str, detail: str = "", failed: bool = False) -> None:
        self._db.execute(
            "UPDATE tasks SET status=?, progress=?, detail=?, finished_at=? WHERE key=?",
            ("failed" if failed else "done", 0.0 if failed else 1.0, detail, now_iso(), key),
        )
        self._db.commit()

    def think(self, text: str, kind: str = "thought", data: dict | None = None) -> int:
        if kind not in THOUGHT_KINDS:
            raise ValueError(f"Unknown thought kind: {kind}")
        cur = self._db.execute(
            "INSERT INTO thoughts (ts, kind, text, data) VALUES (?, ?, ?, ?)",
            (now_iso(), kind, text, json.dumps(data) if data else None),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def record_exam(self, exam: dict) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO exams VALUES (:year, :n_calls, :accuracy, :baseline_up, :trades, "
            ":win_rate, :profit_factor, :stan_return, :hold_return, :passed, :beat_hold)",
            {**exam, "passed": int(exam["passed"]), "beat_hold": int(exam["beat_hold"])},
        )
        self._db.commit()

    def add_lesson(self, kind: str, title: str, text: str, evidence: dict | None = None) -> None:
        self._db.execute(
            "INSERT INTO lessons (ts, kind, title, text, evidence) VALUES (?, ?, ?, ?, ?)",
            (now_iso(), kind, title, text, json.dumps(evidence) if evidence else None),
        )
        self._db.commit()

    # ----- reading -----
    def meta(self) -> dict:
        return {r["key"]: json.loads(r["value"]) for r in self._db.execute("SELECT * FROM meta")}

    def tasks(self) -> list[dict]:
        return [dict(r) for r in self._db.execute("SELECT * FROM tasks ORDER BY id")]

    def thoughts(self, after: int = 0, limit: int = 200) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM thoughts WHERE id > ? ORDER BY id DESC LIMIT ?", (after, limit)
        ).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            d["data"] = json.loads(d["data"]) if d["data"] else None
            out.append(d)
        return out

    def exams(self) -> list[dict]:
        out = []
        for r in self._db.execute("SELECT * FROM exams ORDER BY year"):
            d = dict(r)
            d["passed"], d["beat_hold"] = bool(d["passed"]), bool(d["beat_hold"])
            out.append(d)
        return out

    def lessons(self) -> list[dict]:
        out = []
        for r in self._db.execute("SELECT * FROM lessons ORDER BY id"):
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"]) if d["evidence"] else None
            out.append(d)
        return out
