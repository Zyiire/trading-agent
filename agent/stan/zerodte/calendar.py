"""Major scheduled news days, so the bot can stand aside on them (the spec: avoid major news days).

Dates come from the official pages, copied on 2026-11-16:
  CPI          https://www.bls.gov/bls/news-release/cpi.htm and /schedule/news_release/cpi.htm
  Jobs report  https://www.bls.gov/bls/news-release/empsit.htm and /schedule/news_release/empsit.htm
  FOMC         https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
               (the decision and press conference are on the second day, at 2:00 and 2:30 PM)

CPI and the jobs report come out at 8:30 AM, before the open; FOMC decisions land mid-session.
October 2025's CPI and jobs reports were never published (the 2025 shutdown). 2027 FOMC dates are
tentative on the Fed's page. Outside 2021 to 2027 the calendar doesn't know, and says so (None).
"""
from __future__ import annotations

from datetime import date

FIRST_YEAR, LAST_YEAR = 2021, 2027

CPI = """
2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13
2021-11-10 2021-12-10 2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13
2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13 2023-01-12 2023-02-14 2023-03-14 2023-04-12
2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12 2024-01-11
2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10
2024-11-13 2024-12-11 2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15
2025-08-12 2025-09-11 2025-10-24 2025-12-18 2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12
2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14 2026-11-10 2026-12-10 2027-01-13 2027-02-11
2027-03-10 2027-04-13 2027-05-12 2027-06-10 2027-07-14 2027-08-11 2027-09-14 2027-10-14 2027-11-10
2027-12-10
"""

JOBS = """
2021-02-05 2021-03-05 2021-04-02 2021-05-07 2021-06-04 2021-07-02 2021-08-06 2021-09-03 2021-10-08
2021-11-05 2021-12-03 2022-01-07 2022-02-04 2022-03-04 2022-04-01 2022-05-06 2022-06-03 2022-07-08
2022-08-05 2022-09-02 2022-10-07 2022-11-04 2022-12-02 2023-01-06 2023-02-03 2023-03-10 2023-04-07
2023-05-05 2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08 2024-01-05
2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06 2024-10-04
2024-11-01 2024-12-06 2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03
2025-08-01 2025-09-05 2025-11-20 2025-12-16 2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08
2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02 2026-11-06 2026-12-04 2027-01-08 2027-02-05
2027-03-05 2027-04-02 2027-05-07 2027-06-04 2027-07-02 2027-08-06 2027-09-03 2027-10-08 2027-11-05
2027-12-03
"""

# Decision days (the second day of each meeting).
FOMC = """
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09
2027-01-27 2027-03-17 2027-04-28 2027-06-09 2027-07-28 2027-09-15 2027-10-27 2027-11-17
"""


def _days(text: str) -> frozenset[date]:
    return frozenset(date.fromisoformat(s) for s in text.split())


EVENTS: dict[str, frozenset[date]] = {"CPI": _days(CPI), "jobs report": _days(JOBS), "FOMC": _days(FOMC)}


def events_on(day: date) -> list[str] | None:
    """The big scheduled events on `day` ([] for none), or None when the calendar doesn't cover it."""
    if not FIRST_YEAR <= day.year <= LAST_YEAR:
        return None
    return [name for name, days in EVENTS.items() if day in days]


def is_news_day(day: date) -> bool | None:
    found = events_on(day)
    return None if found is None else bool(found)


def next_events(after: date, n: int = 5) -> list[tuple[date, str]]:
    rows = sorted((d, name) for name, days in EVENTS.items() for d in days if d >= after)
    return rows[:n]
