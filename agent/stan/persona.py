"""Stan: the agent's identity and voice.

Stan's "thoughts" are plain-English lines built from the real numbers his
training produces. Nothing here invents results; if a number is bad, Stan says so.
"""
from __future__ import annotations

STAN = {
    "name": "Stan",
    "title": "Trader in training",
    "goal": "Become a top-tier day trader, one honest test at a time.",
    "bio": (
        "Stan studies market history, sits a test on every year he has not seen, "
        "and keeps a journal of what his wins and losses teach him. He trades "
        "paper money until he earns real money."
    ),
    "principles": [
        "No emotions. Only rules and evidence.",
        "Every loss gets written down and studied.",
        "A result only counts on data I have never seen.",
        "Beating a coin flip is not enough. I have to beat simply holding.",
        "Protect the account first. Profit second.",
    ],
}

# Ranks are earned by passing gates from the build spec, never by time served.
RANKS = [
    ("student", "Student", "Studying history. No trading yet."),
    ("paper_candidate", "Paper Trader (candidate)", "Passed school. Ready to trade paper money."),
    ("paper_trader", "Paper Trader", "Trading paper money against the go-live checks."),
    ("micro_live", "Micro-live Trader", "Trading the $50 account, one contract at a time."),
    ("top_tier", "Top-tier Trader", "Holding a steady edge with real money."),
]


def rank_for(stage: str) -> dict:
    index = next((i for i, r in enumerate(RANKS) if r[0] == stage), 0)
    key, name, description = RANKS[index]
    next_rank = RANKS[index + 1][1] if index + 1 < len(RANKS) else None
    return {"key": key, "name": name, "description": description,
            "level": index + 1, "of": len(RANKS), "next": next_rank}


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def signed_pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{digits}f}%"
