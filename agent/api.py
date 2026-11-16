"""Read-only API the phone app talks to. (The desktop web app in agent/web.py builds on it.)

Run:  python -m agent.api
Then point the app at http://<this computer's IP>:8000

It only reads Stan's journal. It cannot trade. Set STAN_API_TOKEN in .env to
require a token (recommended on shared Wi-Fi).
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from agent.stan.journal import Journal
from agent.stan.persona import STAN, rank_for
from agent.stan.shadow import ShadowBook, snapshot
from agent.stan.skills import library
from agent.stan.paper import PaperBook, snapshot as paper_snapshot


def create_app(journal_path: str | Path = "data/stan.db", token: str | None = None) -> FastAPI:
    app = FastAPI(title="Stan", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

    def auth(request: Request, authorization: str | None = Header(default=None)) -> None:
        local = request.client is not None and request.client.host in ("127.0.0.1", "::1", "localhost")
        if token and not local and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="Bad or missing token")

    def journal() -> Journal:
        j = Journal(journal_path)
        try:
            yield j
        finally:
            j.close()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/stan", dependencies=[Depends(auth)])
    def stan(j: Journal = Depends(journal)) -> dict:
        meta, tasks = j.meta(), j.tasks()
        active = next((t for t in tasks if t["status"] == "active"), None)
        done = sum(t["status"] == "done" for t in tasks)
        return {
            **STAN,
            "rank": rank_for(meta.get("stage", "student")),
            "run_id": meta.get("run_id"),
            "mode": meta.get("mode", "none"),      # "demo", "real" or "none"
            "state": meta.get("state", "idle"),    # "idle", "training", "finished", "stopped"
            "current_task": active,
            "tasks_done": done,
            "tasks_total": len(tasks),
            "summary": meta.get("summary"),
            "strategy": meta.get("strategy"),
            "daytrade_exam": meta.get("daytrade_exam"),
            "trend_exam": meta.get("trend_exam"),
            "price_action_exam": meta.get("price_action_exam"),
            "portfolio": meta.get("portfolio"),
            "arena": meta.get("arena"),
        }

    @app.get("/tasks", dependencies=[Depends(auth)])
    def tasks(j: Journal = Depends(journal)) -> list[dict]:
        return j.tasks()

    @app.get("/thoughts", dependencies=[Depends(auth)])
    def thoughts(after: int = 0, limit: int = 200, j: Journal = Depends(journal)) -> list[dict]:
        return j.thoughts(after=after, limit=max(1, min(limit, 500)))

    @app.get("/exams", dependencies=[Depends(auth)])
    def exams(j: Journal = Depends(journal)) -> list[dict]:
        return j.exams()

    @app.get("/lessons", dependencies=[Depends(auth)])
    def lessons(j: Journal = Depends(journal)) -> list[dict]:
        return j.lessons()

    @app.get("/paper", dependencies=[Depends(auth)])
    def paper() -> dict:
        book = PaperBook(journal_path)
        try:
            return paper_snapshot(book)
        finally:
            book.close()

    @app.get("/skills", dependencies=[Depends(auth)])
    def skills() -> dict:
        return library()

    @app.get("/shadow", dependencies=[Depends(auth)])
    def shadow() -> dict:
        book = ShadowBook(journal_path)
        try:
            return snapshot(book)
        finally:
            book.close()

    return app


def main() -> None:
    import uvicorn

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    app = create_app(os.getenv("STAN_JOURNAL", "data/stan.db"), os.getenv("STAN_API_TOKEN") or None)
    uvicorn.run(app, host=os.getenv("STAN_API_HOST", "0.0.0.0"), port=int(os.getenv("STAN_API_PORT", "8000")))


if __name__ == "__main__":
    main()
