# Day Trader skill pack (vendored)

These folders are an unmodified copy of **The Day Trader skill** (Trader skill v2)
by Jake Schincariol, MIT License (see `LICENSE` in this folder).

- Source: https://github.com/Jakeschincariol/day-trader-skill
- Version: commit `3d0f4ef` (2026-10-20), "The Trader skill v2: 11 day-trading skills"
- Copied: 2026-11-15. The pack's own offline test suite (315 tests) passed on this copy's source.

Stan loads these tools as-is (see `agent/stan/daytrade.py` and `agent/stan/skills.py`).
Do not edit files here. To update, copy a newer release over this folder and
re-run the tests. `tests/fixtures/day_trader/TSLA-60d5m.json.gz` is the pack's own
test fixture (real Yahoo 5-minute bars for TSLA, 60 days to 2026-10-19), MIT licensed.
