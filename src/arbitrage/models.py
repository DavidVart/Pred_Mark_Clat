"""Data models for the arbitrage pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Platform = Literal["polymarket", "kalshi"]
Side = Literal["yes", "no"]


class MarketPair(BaseModel):
    """A mapping between one Polymarket condition and one Kalshi market.

    Both must resolve identically — that is, if Polymarket says YES, Kalshi
    must also say YES. Market pair integrity is the #1 safety property of
    the arb pipeline. A misaligned pair = guaranteed loss on disagreement.
    """

    pair_id: str  # stable key, e.g. "btc-100k-2026-04-30"
    description: str
    polymarket_market_id: str
    kalshi_ticker: str
    category: str = ""  # "crypto" / "sports" / "politics" / "fed"
    expected_resolution_date: datetime | None = None
    # True if both platforms resolve on the same explicit criterion
    # (e.g., CoinGecko close price) and not on human/UMA judgment.
    mechanical_resolution: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: str = ""

    @property
    def is_safe(self) -> bool:
        """Conservative guard for whether to trade this pair at all."""
        return self.mechanical_resolution


class ArbOpportunity(BaseModel):
    """A detected price-dislocation that MIGHT be tradeable after fees.

    Two legal structures:
      - "buy_yes_poly_no_kalshi": buy YES on Polymarket + NO on Kalshi
      - "buy_no_poly_yes_kalshi": buy NO on Polymarket + YES on Kalshi

    We pick the direction with the cheapest basket.
    """

    pair_id: str
    # Side names are from the buyer's perspective.
    poly_side: Side
    poly_price: float
    kalshi_side: Side
    kalshi_price: float
    basket_cost: float  # poly_price + kalshi_price
    gross_spread: float  # 1 - basket_cost  (profit per $1 notional BEFORE fees)
    estimated_fees_pct: float = 0.0  # round-trip fees as fraction of notional
    net_spread: float = 0.0  # gross_spread - estimated_fees_pct
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_profitable(self) -> bool:
        return self.net_spread > 0

    def describe(self) -> str:
        return (
            f"{self.pair_id}: buy {self.poly_side}@${self.poly_price:.3f} poly + "
            f"{self.kalshi_side}@${self.kalshi_price:.3f} kalshi = "
            f"${self.basket_cost:.3f} (gross {self.gross_spread:+.3f}, net {self.net_spread:+.3f})"
        )


class ArbLeg(BaseModel):
    """One of the two sides of an arb position."""

    platform: Platform
    market_id: str   # polymarket condition_id / CLOB token OR kalshi ticker
    side: Side
    price: float
    size: float      # shares / contracts
    cost: float      # price * size
    order_id: str | None = None
    filled: bool = False
    fill_price: float | None = None
    fill_size: float | None = None


class ArbPosition(BaseModel):
    """An open arb — two legs that jointly lock in a payoff at resolution."""

    arb_id: str
    pair_id: str
    poly_leg: ArbLeg
    kalshi_leg: ArbLeg
    entered_at: datetime = Field(default_factory=datetime.utcnow)
    basket_cost: float
    expected_profit: float  # $1 payout - basket_cost - fees
    is_paper: bool = True
    is_complete: bool = False  # both legs filled
    unwind_needed: bool = False  # one filled, the other didn't

    @property
    def notional(self) -> float:
        return self.poly_leg.size  # both legs must have same size
