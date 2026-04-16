"""Abstract exchange client protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.models.market import OrderBook, UnifiedMarket
from src.models.trade import Position


@runtime_checkable
class ExchangeClient(Protocol):
    """Unified interface for prediction market exchanges."""

    @property
    def platform_name(self) -> str: ...

    async def get_active_markets(self, limit: int = 100) -> list[UnifiedMarket]: ...

    async def get_market(self, market_id: str) -> UnifiedMarket | None: ...

    async def get_orderbook(self, market_id: str) -> OrderBook: ...

    async def place_order(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float,
    ) -> str:
        """Place a limit order. Returns order ID."""
        ...

    async def cancel_order(self, order_id: str) -> bool: ...

    async def get_balance(self) -> float: ...

    async def get_positions(self) -> list[Position]: ...

    async def close(self) -> None: ...
