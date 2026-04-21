"""Classify Polymarket BTC market titles by resolution type.

Different market types need DIFFERENT fair-value calculations:

    UP_OR_DOWN   "Bitcoin Up or Down - April 21, 8AM ET"
                 → strike = BTC spot at window open
                 → fair_yes = Φ(log(spot_now / strike) / σ)

    FIXED_STRIKE "Will the price of Bitcoin be above $72,000 on April 21?"
                 → strike = parsed dollar amount ($72,000)
                 → fair_yes = Φ(log(spot_now / strike) / σ)

    BETWEEN      "Will the price of Bitcoin be between $74,000 and $76,000 on April 21?"
                 → strikes = (low, high)
                 → fair_yes = Φ(log(high / spot_now) / σ) − Φ(log(low / spot_now) / σ)
                 Currently UNSUPPORTED — scanner skips these.

This module is import-safe (no I/O, no side effects). Just regex + dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MarketType = Literal["up_or_down", "fixed_strike", "between", "unknown"]


# --- Regexes ---

# "Bitcoin Up or Down - April 21, 8AM ET"
_UP_OR_DOWN_RX = re.compile(r"bitcoin.*up\s*or\s*down", re.IGNORECASE)

# "Will the price of Bitcoin be above $72,000 on April 21?"
# Also handles "hit $150k", "over $100,000", "below $60000"
_FIXED_STRIKE_RX = re.compile(
    r"(?:bitcoin|btc)\b.*?(?:above|below|higher\s+than|over|under|hit|reach|\bbe\s+(?:above|below))"
    r"\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(k|K)?",
    re.IGNORECASE,
)

# "between $74,000 and $76,000"
_BETWEEN_RX = re.compile(
    r"between\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(k|K)?\s*and\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(k|K)?",
    re.IGNORECASE,
)

# "$74-$76k" shorthand — less common but possible
_DASH_RANGE_RX = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*-\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(k|K)?",
)


@dataclass(frozen=True)
class MarketClassification:
    """Parsed market type + strike data."""

    type: MarketType
    # For fixed_strike: the strike dollar amount (e.g., 72000)
    strike: float | None = None
    # For between: low and high strike bounds
    strike_low: float | None = None
    strike_high: float | None = None
    # Direction for fixed_strike: "above" (YES if BTC > strike) or "below"
    direction: Literal["above", "below", "unknown"] = "unknown"

    @property
    def is_supported(self) -> bool:
        """True if NewsAlpha's fair-value model can price this market."""
        return self.type in ("up_or_down", "fixed_strike")


def _parse_dollar_amount(num_str: str, k_flag: str | None) -> float | None:
    """Parse "$68,000" or "150k" into a float dollar amount."""
    try:
        val = float(num_str.replace(",", ""))
    except ValueError:
        return None
    if k_flag and k_flag.lower() == "k":
        val *= 1000
    return val


def classify_title(title: str) -> MarketClassification:
    """Detect market type and extract strike(s) from the question title.

    Returns MarketClassification with type="unknown" if we can't parse it —
    the scanner then skips the market rather than trade it with wrong strike.
    """
    # Order matters: check "between" first because it can also match "above/below"
    # regex via "above X and below Y" wording variants.
    m_between = _BETWEEN_RX.search(title)
    if m_between:
        low = _parse_dollar_amount(m_between.group(1), m_between.group(2))
        high = _parse_dollar_amount(m_between.group(3), m_between.group(4))
        if low is not None and high is not None and low < high:
            return MarketClassification(
                type="between", strike_low=low, strike_high=high
            )

    # Up or Down — explicit marker
    if _UP_OR_DOWN_RX.search(title):
        return MarketClassification(type="up_or_down")

    # Fixed-strike ("above $X" / "below $X")
    m_strike = _FIXED_STRIKE_RX.search(title)
    if m_strike:
        val = _parse_dollar_amount(m_strike.group(1), m_strike.group(2))
        if val is not None:
            direction: Literal["above", "below", "unknown"] = "above"
            lower_title = title.lower()
            if "below" in lower_title or "under" in lower_title:
                direction = "below"
            return MarketClassification(
                type="fixed_strike", strike=val, direction=direction
            )

    return MarketClassification(type="unknown")
