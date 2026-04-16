"""Unified market models that normalize both Polymarket and Kalshi."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UnifiedMarket(BaseModel):
    """Platform-agnostic market representation."""

    platform: Literal["polymarket", "kalshi"]
    market_id: str
    title: str
    description: str = ""
    category: str = ""
    yes_price: float = Field(ge=0.0, le=1.0)
    no_price: float = Field(ge=0.0, le=1.0)
    volume: int = 0
    liquidity: float = 0.0
    expiration: datetime | None = None
    status: str = "active"
    outcomes: list[str] = Field(default_factory=lambda: ["Yes", "No"])
    url: str = ""

    # Polymarket-specific
    clob_token_ids: list[str] | None = None
    condition_id: str | None = None

    # Kalshi-specific
    ticker: str | None = None

    @property
    def spread(self) -> float:
        """Bid-ask spread as fraction of 1."""
        return abs(1.0 - self.yes_price - self.no_price)

    @property
    def mid_price(self) -> float:
        return self.yes_price

    @property
    def time_to_expiry_hours(self) -> float | None:
        if self.expiration is None:
            return None
        # Normalize to TZ-naive UTC so we can subtract utcnow() regardless of input format
        exp = self.expiration.replace(tzinfo=None) if self.expiration.tzinfo else self.expiration
        delta = exp - datetime.utcnow()
        return max(delta.total_seconds() / 3600, 0)


class MarketSnapshot(BaseModel):
    """Timestamped price record for historical tracking."""

    market_id: str
    platform: str
    yes_price: float
    no_price: float
    volume: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OrderBook(BaseModel):
    """Simplified orderbook representation."""

    market_id: str
    bids: list[tuple[float, float]] = Field(default_factory=list)  # (price, size)
    asks: list[tuple[float, float]] = Field(default_factory=list)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
