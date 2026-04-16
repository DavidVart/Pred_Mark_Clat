"""Cross-platform spread scanner.

For each active market pair, fetches current prices from both platforms and
computes the two possible synthetic basket costs:

    Option A: buy YES on Polymarket + NO on Kalshi
              basket_A = poly.yes + kalshi.no

    Option B: buy NO on Polymarket + YES on Kalshi
              basket_B = poly.no + kalshi.yes

The minimum of these two baskets, if less than 1.0, is a gross arbitrage.
After subtracting round-trip fees and a slippage buffer, we get the tradeable
net spread.

We do NOT trade against the mid price — we use the best *ask* price on each
leg because that's what we'd actually pay to cross the spread on entry.
Without a maker bot, taker prices are the realistic assumption.
"""

from __future__ import annotations

import asyncio

from src.arbitrage.matcher import MarketPairRegistry
from src.arbitrage.models import ArbOpportunity, MarketPair
from src.clients.kalshi_client import KalshiClient
from src.clients.polymarket_client import PolymarketClient
from src.pipeline.fees import estimate_round_trip_fee
from src.utils.logging import get_logger

logger = get_logger("arb.scanner")


class ArbScanner:
    def __init__(
        self,
        poly: PolymarketClient,
        kalshi: KalshiClient,
        registry: MarketPairRegistry,
        min_net_spread: float = 0.005,  # 0.5% minimum net profit per $1
        slippage_buffer: float = 0.002,  # extra safety margin
    ):
        self.poly = poly
        self.kalshi = kalshi
        self.registry = registry
        self.min_net_spread = min_net_spread
        self.slippage_buffer = slippage_buffer

    async def scan(self) -> list[ArbOpportunity]:
        """Scan all active pairs and return any profitable opportunities."""
        pairs = await self.registry.list_active()
        if not pairs:
            return []

        logger.info("arb_scan_start", pairs=len(pairs))

        # Fetch prices concurrently
        tasks = [self._quote_pair(p) for p in pairs]
        quotes = await asyncio.gather(*tasks, return_exceptions=True)

        opportunities: list[ArbOpportunity] = []
        for pair, quote in zip(pairs, quotes):
            if isinstance(quote, Exception):
                logger.warning("quote_failed", pair=pair.pair_id, error=str(quote))
                continue
            if quote is None:
                continue

            opp = self._best_opportunity(pair, quote)
            if opp is None:
                continue

            logger.info(
                "arb_checked",
                pair=pair.pair_id,
                basket=f"${opp.basket_cost:.3f}",
                gross=f"{opp.gross_spread:+.3f}",
                net=f"{opp.net_spread:+.3f}",
                profitable=opp.is_profitable,
            )

            if opp.is_profitable:
                opportunities.append(opp)

        logger.info(
            "arb_scan_complete",
            pairs=len(pairs),
            opportunities=len(opportunities),
        )
        return opportunities

    async def _quote_pair(self, pair: MarketPair) -> dict | None:
        """Fetch current prices for a pair. Returns {poly_yes, poly_no, kal_yes, kal_no}."""
        poly_market, kalshi_market = await asyncio.gather(
            self.poly.get_market(pair.polymarket_market_id),
            self.kalshi.get_market(pair.kalshi_ticker),
            return_exceptions=True,
        )

        if isinstance(poly_market, Exception) or poly_market is None:
            logger.debug("poly_quote_missing", pair=pair.pair_id)
            return None
        if isinstance(kalshi_market, Exception) or kalshi_market is None:
            logger.debug("kalshi_quote_missing", pair=pair.pair_id)
            return None

        return {
            "poly_yes": poly_market.yes_price,
            "poly_no": poly_market.no_price,
            "kalshi_yes": kalshi_market.yes_price,
            "kalshi_no": kalshi_market.no_price,
            "poly_market": poly_market,
            "kalshi_market": kalshi_market,
        }

    def _best_opportunity(self, pair: MarketPair, quote: dict) -> ArbOpportunity | None:
        """Compute both synthetic baskets, return the cheaper one as an Opportunity.

        Returns None if the basket is sensible but unprofitable (so the caller
        still gets the data for logging); caller checks .is_profitable.
        """
        # Skip unsafe pairs defensively even if they're in the registry
        if not pair.is_safe:
            logger.debug("pair_unsafe_skipped", pair_id=pair.pair_id)
            return None

        basket_a = quote["poly_yes"] + quote["kalshi_no"]
        basket_b = quote["poly_no"] + quote["kalshi_yes"]

        if basket_a <= basket_b:
            basket = basket_a
            poly_side = "yes"
            poly_price = quote["poly_yes"]
            kalshi_side = "no"
            kalshi_price = quote["kalshi_no"]
        else:
            basket = basket_b
            poly_side = "no"
            poly_price = quote["poly_no"]
            kalshi_side = "yes"
            kalshi_price = quote["kalshi_yes"]

        gross_spread = 1.0 - basket

        # Estimate round-trip fees on BOTH legs. Hold to resolution =
        # winning leg pays $1 with no exit fee; losing leg expires worthless.
        poly_fees = estimate_round_trip_fee(
            "polymarket",
            pair.category,
            poly_price,
            is_maker_entry=False,  # arb MUST cross spread — taker fees apply
            settles_at_resolution=True,
        )
        kalshi_fees = estimate_round_trip_fee(
            "kalshi",
            pair.category,
            kalshi_price,
            is_maker_entry=False,
            settles_at_resolution=True,
        )
        total_fees = poly_fees.round_trip_pct + kalshi_fees.round_trip_pct

        net_spread = gross_spread - total_fees - self.slippage_buffer

        return ArbOpportunity(
            pair_id=pair.pair_id,
            poly_side=poly_side,
            poly_price=poly_price,
            kalshi_side=kalshi_side,
            kalshi_price=kalshi_price,
            basket_cost=basket,
            gross_spread=gross_spread,
            estimated_fees_pct=total_fees + self.slippage_buffer,
            net_spread=net_spread,
        )
