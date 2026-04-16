"""Platform-specific fee models used to compute net edge and net PnL.

Fee references (April 2026):
- Polymarket international CLOB: maker 0%, taker 0.75–1.80% by category
  https://docs.polymarket.com/trading/fees
- Polymarket US DCM: maker -0.20% (rebate), taker 0.30%
  https://www.polymarketexchange.com/fees-hours.html
- Kalshi: taker = ceil(0.07 * C * P * (1-P)) cents per contract; no maker fee on most
  https://kalshi.com/fee-schedule

We intentionally model fees on the CONSERVATIVE side — overestimating fees means
our edge filter is stricter, which is safer than under-filtering on tight trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Polymarket international taker fees by category (fraction of notional)
# Source: docs.polymarket.com/trading/fees as of April 2026
POLYMARKET_TAKER_FEES: dict[str, float] = {
    "sports": 0.0075,
    "politics": 0.0100,
    "economics": 0.0100,
    "tech": 0.0100,
    "crypto": 0.0180,
    "geopolitics": 0.0100,
    "culture": 0.0100,
    "entertainment": 0.0100,
}
POLYMARKET_TAKER_DEFAULT = 0.0125  # middle-of-range if category unknown
POLYMARKET_MAKER_FEE = 0.0          # international CLOB


def polymarket_taker_fee_pct(category: str) -> float:
    return POLYMARKET_TAKER_FEES.get((category or "").lower(), POLYMARKET_TAKER_DEFAULT)


def kalshi_taker_fee_per_contract(price: float) -> float:
    """Kalshi takes ceil(0.07 * 100 * P * (1-P)) cents per 100 contracts.

    Returned as dollars-per-contract fraction of notional.
    """
    if price <= 0 or price >= 1:
        return 0.0
    # Fee in cents per contract (they round up to nearest cent).
    # For a single contract: math.ceil(7 * P * (1-P)) cents, divided by 100 = dollars.
    fee_cents = math.ceil(7.0 * price * (1.0 - price))
    return fee_cents / 100.0  # dollars per contract


@dataclass(frozen=True)
class FeeEstimate:
    entry_fee_pct: float  # fraction of notional on entry
    exit_fee_pct: float   # fraction of notional on expected exit
    round_trip_pct: float  # sum — to be subtracted from gross edge
    notes: str = ""


def estimate_round_trip_fee(
    platform: str,
    category: str,
    price: float,
    is_maker_entry: bool = False,
    settles_at_resolution: bool = False,
) -> FeeEstimate:
    """Worst-case estimate of total fees for a round-trip trade.

    Args:
        platform: "polymarket" or "kalshi"
        category: e.g. "sports", "crypto", "politics"
        price: entry market price in [0,1]
        is_maker_entry: True if entry is a post-only limit at our edge price
        settles_at_resolution: True if we hold to resolution (no exit fee)

    Returns:
        FeeEstimate with entry, exit, and round-trip fractions.
    """
    if platform == "polymarket":
        entry = POLYMARKET_MAKER_FEE if is_maker_entry else polymarket_taker_fee_pct(category)
        # Exit: if we hold to resolution on Polymarket, winning side gets $1 with no exit fee.
        # If we close early, we're likely a taker crossing the spread.
        exit_ = 0.0 if settles_at_resolution else polymarket_taker_fee_pct(category)
        return FeeEstimate(
            entry_fee_pct=entry,
            exit_fee_pct=exit_,
            round_trip_pct=entry + exit_,
            notes=f"Polymarket {category} maker={is_maker_entry} resolves={settles_at_resolution}",
        )

    elif platform == "kalshi":
        # Kalshi charges fees per-contract regardless of maker/taker on most markets.
        # Entry fee as fraction: fee_per_contract / price_per_contract.
        per_contract = kalshi_taker_fee_per_contract(price)
        entry = per_contract / price if price > 0 else 0.0
        # If holding to resolution on Kalshi, no exit fee (winners settle to $1).
        exit_ = 0.0 if settles_at_resolution else (
            kalshi_taker_fee_per_contract(price) / price if price > 0 else 0.0
        )
        return FeeEstimate(
            entry_fee_pct=entry,
            exit_fee_pct=exit_,
            round_trip_pct=entry + exit_,
            notes=f"Kalshi per-contract={per_contract:.4f} at price={price:.2f}",
        )

    # Unknown platform — be conservative
    return FeeEstimate(
        entry_fee_pct=0.02,
        exit_fee_pct=0.02,
        round_trip_pct=0.04,
        notes=f"Unknown platform {platform}, using conservative 4% round-trip",
    )


def net_edge(
    gross_edge: float,
    platform: str,
    category: str,
    price: float,
    is_maker_entry: bool = False,
    settles_at_resolution: bool = True,
) -> tuple[float, FeeEstimate]:
    """Convert gross edge to net edge after fees.

    Defaults assume we hold to resolution (common for directional LLM bets).
    Returns (net_edge, fee_breakdown).
    """
    fees = estimate_round_trip_fee(
        platform=platform,
        category=category,
        price=price,
        is_maker_entry=is_maker_entry,
        settles_at_resolution=settles_at_resolution,
    )
    # Edge is a probability differential; fees eat into the payout.
    # A gross edge of +0.10 with 2% round-trip fees becomes ~+0.08 net.
    return gross_edge - fees.round_trip_pct, fees
