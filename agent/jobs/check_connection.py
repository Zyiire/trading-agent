"""Step 1: confirm the agent can see your Alpaca paper account.

Run:  python -m agent.jobs.check_connection
It only reads. It cannot place, change or cancel anything.
"""
from __future__ import annotations

import sys

from agent.broker.alpaca import AlpacaBroker
from agent.config import ConfigError, load_settings
from agent.store import Store


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as err:
        print(f"Setup problem: {err}")
        return 1

    broker = AlpacaBroker(settings)
    try:
        account = broker.get_account()
        clock = broker.get_clock()
        positions = broker.get_positions()
    except Exception as err:  # network or bad keys
        print(f"Could not reach Alpaca: {type(err).__name__}: {err}")
        print("Check that the keys in .env are your PAPER keys and that you are online.")
        return 1

    print("Connected to Alpaca (PAPER account)")
    print(f"  Status:               {account.status}")
    print(f"  Equity:               ${account.equity:,.2f}")
    print(f"  Cash:                 ${account.cash:,.2f}")
    print(f"  Buying power:         ${account.buying_power:,.2f}")
    if account.options_buying_power is not None:
        print(f"  Options buying power: ${account.options_buying_power:,.2f}")
    print(f"  Options level:        {account.options_level if account.options_level is not None else 'unknown'}")
    print(f"  Account type:         {'cash' if account.multiplier == '1' else 'margin (multiplier ' + str(account.multiplier) + ')'}")
    print(f"  Market open now:      {'yes' if clock.is_open else 'no'} (next open {clock.next_open:%a %b %d %H:%M %Z})")
    print(f"  Open positions:       {len(positions)}")
    for p in positions:
        print(f"    {p.symbol}: {p.qty:g} @ ${p.avg_entry_price:,.2f}")

    warnings = []
    if account.trading_blocked:
        warnings.append("Trading is blocked on this account.")
    if not account.options_level:
        warnings.append("Options trading is not enabled. Turn it on in the Alpaca dashboard for the paper account.")
    if account.multiplier not in (None, "1"):
        warnings.append("This is a margin account. The spec calls for a cash account to avoid the Pattern Day Trader rule.")
    if abs(account.equity - settings.starting_capital) > 0.01:
        warnings.append(
            f"Paper equity is ${account.equity:,.2f}, not the ${settings.starting_capital:,.2f} in the plan. "
            "Reset the paper account balance in the Alpaca dashboard, or the agent will size trades off the larger balance."
        )
    if warnings:
        print("\nThings to fix:")
        for w in warnings:
            print(f"  - {w}")

    store = Store(settings.db_path)
    store.save_account_snapshot(account)
    store.close()
    print(f"\nKill switch level for this plan: ${settings.kill_switch_equity:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
