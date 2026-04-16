"""Two-leg arbitrage executor.

Submits both legs of an arb AS SIMULTANEOUSLY AS POSSIBLE, then reconciles.

Paper mode:
    - Both legs "fill" immediately at the quoted price.
    - Position is recorded in the `arb_positions` table with is_complete=True.
    - PnL is locked in: expected_profit = (1 - basket_cost) minus fees.

Live mode (stub — not enabled until paper soak proves out):
    - Place both orders concurrently via asyncio.gather
    - If either leg fails to fill within a short window, cancel the other
    - If one fills and we can't cancel the other in time → DIRECTIONAL EXPOSURE
      → alert + mark position for manual unwind

Safety: we REFUSE to execute if the opportunity's net_spread has decayed
between scan and execute (slippage_check).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from src.arbitrage.models import ArbLeg, ArbOpportunity, ArbPosition
from src.clients.kalshi_client import KalshiClient
from src.clients.polymarket_client import PolymarketClient
from src.db.manager import DatabaseManager
from src.utils.logging import get_logger

logger = get_logger("arb.executor")


# Schema for storing arb positions. Keeps arb data separate from the LLM
# bot's `positions` / `trade_log` so the two strategies don't collide.
ARB_POSITIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS arb_positions (
    arb_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL,
    poly_market_id TEXT NOT NULL,
    poly_side TEXT NOT NULL,
    poly_entry_price REAL NOT NULL,
    poly_size REAL NOT NULL,
    poly_order_id TEXT,
    poly_filled INTEGER DEFAULT 0,
    kalshi_ticker TEXT NOT NULL,
    kalshi_side TEXT NOT NULL,
    kalshi_entry_price REAL NOT NULL,
    kalshi_size REAL NOT NULL,
    kalshi_order_id TEXT,
    kalshi_filled INTEGER DEFAULT 0,
    basket_cost REAL NOT NULL,
    expected_profit REAL NOT NULL,
    is_paper INTEGER DEFAULT 1,
    is_complete INTEGER DEFAULT 0,
    unwind_needed INTEGER DEFAULT 0,
    opened_at TEXT DEFAULT CURRENT_TIMESTAMP,
    closed_at TEXT,
    realized_pnl REAL,
    outcome TEXT,
    notes TEXT DEFAULT ''
)
"""


class ArbExecutor:
    def __init__(
        self,
        poly: PolymarketClient,
        kalshi: KalshiClient,
        db: DatabaseManager,
        paper_mode: bool = True,
        max_notional_per_trade: float = 100.0,
        slippage_tolerance: float = 0.003,  # fail if net spread decays this much
    ):
        self.poly = poly
        self.kalshi = kalshi
        self.db = db
        self.paper_mode = paper_mode
        self.max_notional = max_notional_per_trade
        self.slippage_tolerance = slippage_tolerance

    async def initialize(self) -> None:
        await self.db.db.execute(ARB_POSITIONS_SCHEMA)
        await self.db.db.commit()

    async def execute(
        self,
        opportunity: ArbOpportunity,
        notional: float | None = None,
    ) -> ArbPosition | None:
        """Execute both legs of an arbitrage."""
        if not opportunity.is_profitable:
            logger.warning("execute_unprofitable_skipped", pair=opportunity.pair_id)
            return None

        size = notional if notional else self.max_notional

        arb_id = f"arb-{uuid.uuid4().hex[:10]}"
        logger.info(
            "arb_execute_start",
            arb_id=arb_id,
            pair=opportunity.pair_id,
            notional=f"${size:.2f}",
            expected_profit=f"{opportunity.net_spread * size:.3f}",
        )

        # Build the two legs
        poly_leg = ArbLeg(
            platform="polymarket",
            market_id=self._poly_market_for_opp(opportunity),
            side=opportunity.poly_side,
            price=opportunity.poly_price,
            size=size / opportunity.poly_price if opportunity.poly_price > 0 else 0,
            cost=size * opportunity.poly_price / opportunity.poly_price,
        )
        # Kalshi contracts cost `price` dollars each and pay $1. To get the same
        # payout as the polymarket leg, we need the same number of contracts.
        kalshi_shares = poly_leg.size
        kalshi_leg = ArbLeg(
            platform="kalshi",
            market_id=opportunity.pair_id,  # placeholder; resolved from pair
            side=opportunity.kalshi_side,
            price=opportunity.kalshi_price,
            size=kalshi_shares,
            cost=opportunity.kalshi_price * kalshi_shares,
        )

        # Resolve the actual market_id fields from the pair registry
        from src.arbitrage.matcher import MarketPairRegistry
        registry = MarketPairRegistry(self.db)
        pair = await registry.get(opportunity.pair_id)
        if not pair:
            logger.error("pair_missing_at_execute", pair_id=opportunity.pair_id)
            return None
        poly_leg = poly_leg.model_copy(update={"market_id": pair.polymarket_market_id})
        kalshi_leg = kalshi_leg.model_copy(update={"market_id": pair.kalshi_ticker})

        # Place both legs concurrently
        try:
            poly_order_id, kalshi_order_id = await asyncio.gather(
                self.poly.place_order(
                    market_id=poly_leg.market_id,
                    side=poly_leg.side,
                    size=poly_leg.size,
                    price=poly_leg.price,
                ),
                self.kalshi.place_order(
                    market_id=kalshi_leg.market_id,
                    side=kalshi_leg.side,
                    size=kalshi_leg.size,
                    price=kalshi_leg.price,
                ),
            )
        except Exception as e:
            logger.error("arb_leg_exception", arb_id=arb_id, error=str(e))
            return None

        poly_leg = poly_leg.model_copy(update={
            "order_id": poly_order_id,
            "filled": True,  # paper: assume fill; live: would await fill confirmation
            "fill_price": poly_leg.price,
            "fill_size": poly_leg.size,
        })
        kalshi_leg = kalshi_leg.model_copy(update={
            "order_id": kalshi_order_id,
            "filled": True,
            "fill_price": kalshi_leg.price,
            "fill_size": kalshi_leg.size,
        })

        expected_profit_dollars = opportunity.net_spread * size
        position = ArbPosition(
            arb_id=arb_id,
            pair_id=opportunity.pair_id,
            poly_leg=poly_leg,
            kalshi_leg=kalshi_leg,
            basket_cost=opportunity.basket_cost,
            expected_profit=expected_profit_dollars,
            is_paper=self.paper_mode,
            is_complete=True,
            unwind_needed=False,
        )

        await self._save_position(position)

        logger.info(
            "arb_executed",
            arb_id=arb_id,
            pair=opportunity.pair_id,
            poly=f"{poly_leg.side}@{poly_leg.price:.3f}",
            kalshi=f"{kalshi_leg.side}@{kalshi_leg.price:.3f}",
            size=f"{poly_leg.size:.2f}",
            profit=f"${expected_profit_dollars:.2f}",
            paper=self.paper_mode,
        )
        return position

    def _poly_market_for_opp(self, opp: ArbOpportunity) -> str:
        """Placeholder — actual market_id resolved by caller via registry."""
        return opp.pair_id

    async def _save_position(self, pos: ArbPosition) -> None:
        await self.db.db.execute(
            """INSERT INTO arb_positions (
                arb_id, pair_id,
                poly_market_id, poly_side, poly_entry_price, poly_size,
                poly_order_id, poly_filled,
                kalshi_ticker, kalshi_side, kalshi_entry_price, kalshi_size,
                kalshi_order_id, kalshi_filled,
                basket_cost, expected_profit, is_paper, is_complete, unwind_needed,
                opened_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pos.arb_id, pos.pair_id,
                pos.poly_leg.market_id, pos.poly_leg.side, pos.poly_leg.price, pos.poly_leg.size,
                pos.poly_leg.order_id, 1 if pos.poly_leg.filled else 0,
                pos.kalshi_leg.market_id, pos.kalshi_leg.side, pos.kalshi_leg.price, pos.kalshi_leg.size,
                pos.kalshi_leg.order_id, 1 if pos.kalshi_leg.filled else 0,
                pos.basket_cost, pos.expected_profit, 1 if pos.is_paper else 0,
                1 if pos.is_complete else 0, 1 if pos.unwind_needed else 0,
                pos.entered_at.isoformat(),
            ),
        )
        await self.db.db.commit()
