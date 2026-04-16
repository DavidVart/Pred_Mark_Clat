"""Stage 1: Market Scanner — discover and filter tradeable markets."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from config.settings import ScannerConfig
from src.clients.base_client import ExchangeClient
from src.db.manager import DatabaseManager
from src.models.market import UnifiedMarket
from src.pipeline.market_filters import classify_market
from src.utils.logging import get_logger

logger = get_logger("scanner")


class MarketScanner:
    """Fetches markets from all platforms and filters for trading candidates."""

    def __init__(
        self,
        clients: list[ExchangeClient],
        config: ScannerConfig,
        db: DatabaseManager,
    ):
        self.clients = clients
        self.config = config
        self.db = db

    async def scan(self) -> list[UnifiedMarket]:
        """Scan all platforms and return filtered, ranked candidates."""
        # Fetch from all platforms concurrently
        tasks = [client.get_active_markets(limit=200) for client in self.clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_markets: list[UnifiedMarket] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                platform = self.clients[i].platform_name
                logger.error("scan_failed", platform=platform, error=str(result))
            else:
                all_markets.extend(result)

        logger.info("scan_fetched", total=len(all_markets))

        # Apply filters
        candidates = []
        for market in all_markets:
            if self._passes_filters(market):
                candidates.append(market)

        # Remove markets on cooldown
        filtered = []
        for market in candidates:
            if not await self.db.is_on_cooldown(market.market_id):
                filtered.append(market)
            else:
                logger.debug("skipped_cooldown", market=market.title[:50])

        # Remove markets we already have positions in
        open_positions = await self.db.get_open_positions()
        position_market_ids = {p["market_id"] for p in open_positions}
        filtered = [m for m in filtered if m.market_id not in position_market_ids]

        # Sort by volume descending (more liquid = better)
        filtered.sort(key=lambda m: m.volume, reverse=True)

        # Take top N candidates
        result = filtered[: self.config.max_candidates]

        logger.info(
            "scan_complete",
            fetched=len(all_markets),
            passed_filters=len(candidates),
            after_cooldown=len(filtered),
            returned=len(result),
        )

        # Store snapshots for historical tracking
        for market in result:
            await self.db.upsert_market(market.model_dump(mode="json"))

        return result

    def _passes_filters(self, market: UnifiedMarket) -> bool:
        """Apply deterministic filters to a market."""
        # Volume filter
        if market.volume < self.config.min_volume:
            return False

        # Price filter — skip near-certain outcomes
        if market.yes_price < self.config.min_price or market.yes_price > self.config.max_price:
            return False

        # Time to expiry filter
        if market.expiration:
            now = datetime.utcnow()
            time_to_expiry = market.expiration.replace(tzinfo=None) - now

            if time_to_expiry < timedelta(hours=self.config.min_time_to_expiry_hours):
                return False
            if time_to_expiry > timedelta(days=self.config.max_time_to_expiry_days):
                return False

        # Must be active
        if market.status not in ("active", "open"):
            return False

        # Resolution-quality filter — skip markets prone to UMA/manual disputes
        if getattr(self.config, "skip_ambiguous_resolution", True):
            quality = classify_market(market)
            if quality.is_ambiguous:
                logger.debug(
                    "skipped_ambiguous",
                    market=market.title[:60],
                    reason=quality.reason,
                )
                return False

        return True
