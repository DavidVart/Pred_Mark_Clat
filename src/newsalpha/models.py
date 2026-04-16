"""Data models for NewsAlpha.

These are deliberately kept separate from the LLM bot's models in src/models/
so the two strategies don't bleed schemas into each other.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Side = Literal["yes", "no"]
TickSource = Literal["coinbase", "binance", "kraken"]


class PriceTick(BaseModel):
    """A single BTC/ETH spot price update from an exchange websocket."""

    symbol: str  # e.g. "BTC-USD"
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: TickSource = "coinbase"


class MarketQuote(BaseModel):
    """A Polymarket prediction-market quote at a moment in time.

    For Polymarket 5-minute BTC markets like "Will BTC close higher at 15:05?".
    yes_price is the implied probability of YES (0-1).
    """

    market_id: str
    title: str
    yes_price: float = Field(ge=0.0, le=1.0)
    no_price: float = Field(ge=0.0, le=1.0)
    window_start: datetime  # opening of the 5M window
    window_end: datetime    # resolution time of the window
    starting_ref_price: float | None = None  # BTC price at window_start (if known)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, (self.window_end - datetime.utcnow()).total_seconds())

    @property
    def seconds_elapsed(self) -> float:
        total = (self.window_end - self.window_start).total_seconds()
        return max(0.0, total - self.seconds_remaining)


class FairValue(BaseModel):
    """Computed fair-value probability for a market, given current BTC price."""

    market_id: str
    fair_yes_probability: float = Field(ge=0.0, le=1.0)
    input_spot_price: float
    input_starting_ref: float
    seconds_remaining: float
    volatility_used: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DivergenceSignal(BaseModel):
    """A detected divergence between a Polymarket market price and our fair value.

    gap = fair_yes - market_yes
    If gap > threshold → BUY YES (market underpricing YES)
    If gap < -threshold → BUY NO (market overpricing YES)
    """

    market_id: str
    title: str
    side: Side
    market_price: float      # current market price on the side we're buying
    fair_value: float         # our computed fair probability for that side
    edge: float               # fair - market (positive = we're right)
    seconds_remaining: float
    spot_reference: float     # current BTC spot
    spot_at_window_start: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def describe(self) -> str:
        return (
            f"{self.title[:50]} | {self.side.upper()} "
            f"market={self.market_price:.3f} fair={self.fair_value:.3f} "
            f"edge={self.edge:+.3f} t-{self.seconds_remaining:.0f}s"
        )
