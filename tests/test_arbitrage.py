"""Tests for the arbitrage pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from src.arbitrage.executor import ArbExecutor, ARB_POSITIONS_SCHEMA
from src.arbitrage.matcher import MarketPairRegistry
from src.arbitrage.models import ArbOpportunity, MarketPair
from src.arbitrage.scanner import ArbScanner
from src.models.market import OrderBook, UnifiedMarket


# --- Mock clients -----------------------------------------------------------

class MockPoly:
    platform_name = "polymarket"

    def __init__(self):
        self.markets: dict[str, UnifiedMarket] = {}
        self.orders = []

    def set_market(self, market_id: str, yes: float, no: float | None = None):
        no = 1.0 - yes if no is None else no
        self.markets[market_id] = UnifiedMarket(
            platform="polymarket",
            market_id=market_id,
            title=f"Poly mock {market_id}",
            yes_price=yes,
            no_price=no,
            volume=500_000,
            expiration=datetime.utcnow() + timedelta(days=5),
        )

    async def get_market(self, market_id: str) -> UnifiedMarket | None:
        return self.markets.get(market_id)

    async def get_orderbook(self, market_id: str) -> OrderBook:
        return OrderBook(market_id=market_id)

    async def place_order(self, market_id: str, side: str, size: float, price: float) -> str:
        oid = f"poly-mock-{len(self.orders) + 1}"
        self.orders.append({"id": oid, "market_id": market_id, "side": side, "size": size, "price": price})
        return oid

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def close(self) -> None:
        pass


class MockKalshi:
    platform_name = "kalshi"

    def __init__(self):
        self.markets: dict[str, UnifiedMarket] = {}
        self.orders = []

    def set_market(self, ticker: str, yes: float, no: float | None = None):
        no = 1.0 - yes if no is None else no
        self.markets[ticker] = UnifiedMarket(
            platform="kalshi",
            market_id=ticker,
            title=f"Kalshi mock {ticker}",
            yes_price=yes,
            no_price=no,
            volume=200_000,
            expiration=datetime.utcnow() + timedelta(days=5),
            ticker=ticker,
        )

    async def get_market(self, market_id: str) -> UnifiedMarket | None:
        return self.markets.get(market_id)

    async def get_orderbook(self, market_id: str) -> OrderBook:
        return OrderBook(market_id=market_id)

    async def place_order(self, market_id: str, side: str, size: float, price: float) -> str:
        oid = f"kalshi-mock-{len(self.orders) + 1}"
        self.orders.append({"id": oid, "market_id": market_id, "side": side, "size": size, "price": price})
        return oid

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def close(self) -> None:
        pass


# --- Fixtures ---------------------------------------------------------------

@pytest_asyncio.fixture
async def registry(db):
    reg = MarketPairRegistry(db)
    await reg.initialize()
    return reg


@pytest.fixture
def poly():
    return MockPoly()


@pytest.fixture
def kalshi():
    return MockKalshi()


@pytest.fixture
def sample_pair() -> MarketPair:
    return MarketPair(
        pair_id="btc-100k-2026-04-30",
        description="BTC close above $100k on Apr 30",
        polymarket_market_id="poly-btc-100k",
        kalshi_ticker="KAL-BTC-100K",
        category="crypto",
        mechanical_resolution=True,
    )


# --- Matcher / Registry -----------------------------------------------------

class TestMarketPairRegistry:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, registry, sample_pair):
        await registry.upsert(sample_pair)
        fetched = await registry.get(sample_pair.pair_id)
        assert fetched is not None
        assert fetched.polymarket_market_id == "poly-btc-100k"

    @pytest.mark.asyncio
    async def test_list_active_excludes_deactivated(self, registry, sample_pair):
        await registry.upsert(sample_pair)
        # Add a second pair
        second = sample_pair.model_copy(update={"pair_id": "second", "polymarket_market_id": "x"})
        await registry.upsert(second)

        active_before = await registry.list_active()
        assert len(active_before) == 2

        await registry.deactivate(sample_pair.pair_id, "test")
        active_after = await registry.list_active()
        assert len(active_after) == 1
        assert active_after[0].pair_id == "second"

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, registry, sample_pair):
        await registry.upsert(sample_pair)
        updated = sample_pair.model_copy(update={"description": "CHANGED"})
        await registry.upsert(updated)
        fetched = await registry.get(sample_pair.pair_id)
        assert fetched.description == "CHANGED"


# --- Scanner ----------------------------------------------------------------

class TestArbScanner:
    @pytest.mark.asyncio
    async def test_detects_profitable_spread(self, registry, poly, kalshi, sample_pair):
        await registry.upsert(sample_pair)
        # Poly yes at 0.50, Kalshi yes at 0.40 → Kalshi cheaper on YES.
        # Buy YES@kalshi (0.40) + NO@poly (0.50) = 0.90 basket → 10% gross spread.
        poly.set_market("poly-btc-100k", yes=0.50)
        kalshi.set_market("KAL-BTC-100K", yes=0.40)

        scanner = ArbScanner(poly, kalshi, registry, min_net_spread=0.005)
        opps = await scanner.scan()
        assert len(opps) == 1
        opp = opps[0]
        assert opp.net_spread > 0
        # Direction should be: buy NO on poly, YES on kalshi
        assert opp.poly_side == "no"
        assert opp.kalshi_side == "yes"

    @pytest.mark.asyncio
    async def test_no_opportunity_when_aligned(self, registry, poly, kalshi, sample_pair):
        await registry.upsert(sample_pair)
        poly.set_market("poly-btc-100k", yes=0.50)
        kalshi.set_market("KAL-BTC-100K", yes=0.50)
        scanner = ArbScanner(poly, kalshi, registry, min_net_spread=0.005)
        opps = await scanner.scan()
        # Basket = 0.50 + 0.50 = 1.00 → zero gross, no opportunity after fees
        assert opps == []

    @pytest.mark.asyncio
    async def test_skips_unsafe_pair(self, registry, poly, kalshi, sample_pair):
        unsafe = sample_pair.model_copy(update={"mechanical_resolution": False})
        await registry.upsert(unsafe)
        poly.set_market("poly-btc-100k", yes=0.30)
        kalshi.set_market("KAL-BTC-100K", yes=0.30)
        scanner = ArbScanner(poly, kalshi, registry)
        opps = await scanner.scan()
        assert opps == []

    @pytest.mark.asyncio
    async def test_ignores_tiny_spread_below_threshold(self, registry, poly, kalshi, sample_pair):
        await registry.upsert(sample_pair)
        # Basket = 0.98 → 2% gross but fees eat most of it
        poly.set_market("poly-btc-100k", yes=0.50)
        kalshi.set_market("KAL-BTC-100K", yes=0.48)
        scanner = ArbScanner(poly, kalshi, registry, min_net_spread=0.005)
        opps = await scanner.scan()
        # Depending on fees, might still be profitable; assert it's bounded
        for o in opps:
            assert o.net_spread >= 0.005


# --- Executor ---------------------------------------------------------------

class TestArbExecutor:
    @pytest.mark.asyncio
    async def test_paper_execute_records_position(self, registry, poly, kalshi, db, sample_pair):
        await registry.upsert(sample_pair)
        poly.set_market("poly-btc-100k", yes=0.50)
        kalshi.set_market("KAL-BTC-100K", yes=0.40)

        scanner = ArbScanner(poly, kalshi, registry, min_net_spread=0.005)
        opps = await scanner.scan()
        assert opps

        executor = ArbExecutor(poly, kalshi, db, paper_mode=True, max_notional_per_trade=100.0)
        await executor.initialize()
        pos = await executor.execute(opps[0], notional=100.0)

        assert pos is not None
        assert pos.is_complete
        assert pos.is_paper
        assert not pos.unwind_needed

        # Check DB
        cursor = await db.db.execute("SELECT COUNT(*) FROM arb_positions")
        row = await cursor.fetchone()
        assert row[0] == 1

        # Mock clients saw orders on BOTH platforms
        assert len(poly.orders) == 1
        assert len(kalshi.orders) == 1

    @pytest.mark.asyncio
    async def test_declines_unprofitable(self, registry, poly, kalshi, db, sample_pair):
        executor = ArbExecutor(poly, kalshi, db, paper_mode=True)
        await executor.initialize()
        # Build an explicitly unprofitable opportunity
        bad = ArbOpportunity(
            pair_id=sample_pair.pair_id,
            poly_side="yes",
            poly_price=0.55,
            kalshi_side="no",
            kalshi_price=0.50,
            basket_cost=1.05,
            gross_spread=-0.05,
            estimated_fees_pct=0.02,
            net_spread=-0.07,
        )
        result = await executor.execute(bad)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_pair_fails_safely(self, registry, poly, kalshi, db):
        executor = ArbExecutor(poly, kalshi, db, paper_mode=True)
        await executor.initialize()
        phantom = ArbOpportunity(
            pair_id="nonexistent-pair",
            poly_side="yes",
            poly_price=0.30,
            kalshi_side="no",
            kalshi_price=0.30,
            basket_cost=0.60,
            gross_spread=0.40,
            estimated_fees_pct=0.02,
            net_spread=0.38,
        )
        result = await executor.execute(phantom)
        assert result is None


# --- Arb Opportunity model --------------------------------------------------

class TestArbOpportunityModel:
    def test_is_profitable_positive(self):
        opp = ArbOpportunity(
            pair_id="x",
            poly_side="yes",
            poly_price=0.3,
            kalshi_side="no",
            kalshi_price=0.3,
            basket_cost=0.6,
            gross_spread=0.4,
            estimated_fees_pct=0.02,
            net_spread=0.38,
        )
        assert opp.is_profitable

    def test_is_profitable_negative(self):
        opp = ArbOpportunity(
            pair_id="x",
            poly_side="yes",
            poly_price=0.5,
            kalshi_side="no",
            kalshi_price=0.5,
            basket_cost=1.0,
            gross_spread=0.0,
            estimated_fees_pct=0.02,
            net_spread=-0.02,
        )
        assert not opp.is_profitable

    def test_describe_has_key_info(self):
        opp = ArbOpportunity(
            pair_id="btc",
            poly_side="yes",
            poly_price=0.30,
            kalshi_side="no",
            kalshi_price=0.40,
            basket_cost=0.70,
            gross_spread=0.30,
            estimated_fees_pct=0.02,
            net_spread=0.28,
        )
        s = opp.describe()
        assert "btc" in s
        assert "poly" in s
        assert "kalshi" in s
