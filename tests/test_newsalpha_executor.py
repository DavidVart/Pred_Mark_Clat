"""Tests for NewsAlpha executor — position opening, exits, Kelly sizing."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from src.newsalpha.db import NewsAlphaDB
from src.newsalpha.executor import NewsAlphaExecutor, NewsAlphaExecutorConfig
from src.newsalpha.models import DivergenceSignal, MarketQuote


@pytest_asyncio.fixture
async def na_db():
    db = NewsAlphaDB(":memory:")
    await db.initialize()
    yield db
    await db.close()


def _signal(
    market="m1",
    side="yes",
    edge=0.05,
    market_price=0.45,
    sec_left=300.0,
    title="Mock BTC Up/Down",
) -> DivergenceSignal:
    return DivergenceSignal(
        market_id=market,
        title=title,
        side=side,
        market_price=market_price,
        fair_value=market_price + edge if side == "yes" else market_price - edge,
        edge=edge,
        seconds_remaining=sec_left,
        spot_reference=74000.0,
        spot_at_window_start=73800.0,
    )


def _quote(market_id="m1", yes=0.50, no=None, sec_left=200.0) -> MarketQuote:
    no = 1.0 - yes if no is None else no
    now = datetime.utcnow()
    return MarketQuote(
        market_id=market_id,
        title="Mock",
        yes_price=yes,
        no_price=no,
        window_start=now - timedelta(seconds=100),
        window_end=now + timedelta(seconds=sec_left),
    )


class TestOpenPosition:
    @pytest.mark.asyncio
    async def test_opens_position_on_signal(self, na_db):
        ex = NewsAlphaExecutor(NewsAlphaExecutorConfig(bankroll=1000), na_db)
        opened = await ex.on_signal(_signal(edge=0.05, sec_left=300))
        assert opened
        assert ex.open_position_count == 1

    @pytest.mark.asyncio
    async def test_rejects_duplicate_market(self, na_db):
        ex = NewsAlphaExecutor(NewsAlphaExecutorConfig(bankroll=1000), na_db)
        await ex.on_signal(_signal(market="m1", side="yes"))
        opened_again = await ex.on_signal(_signal(market="m1", side="yes"))
        assert not opened_again
        assert ex.open_position_count == 1

    @pytest.mark.asyncio
    async def test_side_reversal_closes_and_reopens(self, na_db):
        ex = NewsAlphaExecutor(NewsAlphaExecutorConfig(bankroll=1000), na_db)
        await ex.on_signal(_signal(market="m1", side="yes"))
        opened = await ex.on_signal(_signal(market="m1", side="no"))
        assert opened
        assert ex.open_position_count == 1
        # One trade should be recorded (the closed YES position)
        cursor = await na_db.db.execute("SELECT COUNT(*) FROM na_trades")
        assert (await cursor.fetchone())[0] == 1

    @pytest.mark.asyncio
    async def test_respects_max_positions(self, na_db):
        ex = NewsAlphaExecutor(NewsAlphaExecutorConfig(bankroll=1000, max_positions=2), na_db)
        await ex.on_signal(_signal(market="m1"))
        await ex.on_signal(_signal(market="m2"))
        opened = await ex.on_signal(_signal(market="m3"))
        assert not opened
        assert ex.open_position_count == 2

    @pytest.mark.asyncio
    async def test_rejects_near_resolution(self, na_db):
        cfg = NewsAlphaExecutorConfig(bankroll=1000, flatten_before_resolution_seconds=60.0)
        ex = NewsAlphaExecutor(cfg, na_db)
        # Too close: 100s left, flatten needs 120s (2x 60s)
        opened = await ex.on_signal(_signal(sec_left=100))
        assert not opened


class TestExitLogic:
    @pytest.mark.asyncio
    async def test_stop_loss(self, na_db):
        cfg = NewsAlphaExecutorConfig(bankroll=1000, stop_loss_pct=0.03)
        ex = NewsAlphaExecutor(cfg, na_db)
        await ex.on_signal(_signal(market="m1", side="yes", market_price=0.50, sec_left=600))

        # Price drops from 0.50 → 0.47 → -6% → triggers 3% stop
        exits = await ex.check_exits({"m1": _quote("m1", yes=0.47, sec_left=500)})
        assert exits == 1
        assert ex.open_position_count == 0

        cursor = await na_db.db.execute("SELECT exit_reason FROM na_trades LIMIT 1")
        row = await cursor.fetchone()
        assert row[0] == "stop_loss"

    @pytest.mark.asyncio
    async def test_time_stop(self, na_db):
        cfg = NewsAlphaExecutorConfig(bankroll=1000, flatten_before_resolution_seconds=60.0)
        ex = NewsAlphaExecutor(cfg, na_db)
        await ex.on_signal(_signal(market="m1", sec_left=600))

        # Simulate time passing — quote says only 30s left
        exits = await ex.check_exits({"m1": _quote("m1", yes=0.52, sec_left=30)})
        assert exits == 1

        cursor = await na_db.db.execute("SELECT exit_reason FROM na_trades LIMIT 1")
        row = await cursor.fetchone()
        assert row[0] == "time_stop"

    @pytest.mark.asyncio
    async def test_trailing_lock(self, na_db):
        cfg = NewsAlphaExecutorConfig(
            bankroll=1000,
            trailing_activation_pct=0.02,
            trailing_profit_lock_pct=0.40,
            stop_loss_pct=0.10,
        )
        ex = NewsAlphaExecutor(cfg, na_db)
        await ex.on_signal(_signal(market="m1", side="yes", market_price=0.50, sec_left=600))

        # Price goes UP → 0.55 (+10%) → peak_pnl = 10%
        await ex.check_exits({"m1": _quote("m1", yes=0.55, sec_left=500)})
        assert ex.open_position_count == 1  # still open, trailing not triggered

        # Price pulls back → 0.525 → PnL now +5%, trail level = 10% * (1 - 0.40) = 6%
        # 5% < 6% → triggers trailing lock
        exits = await ex.check_exits({"m1": _quote("m1", yes=0.525, sec_left=450)})
        assert exits == 1

        cursor = await na_db.db.execute("SELECT exit_reason FROM na_trades LIMIT 1")
        row = await cursor.fetchone()
        assert row[0] == "trailing_lock"


class TestKellySizing:
    @pytest.mark.asyncio
    async def test_size_scales_with_edge(self, na_db):
        ex = NewsAlphaExecutor(NewsAlphaExecutorConfig(bankroll=1000), na_db)
        small = ex._compute_size(0.02)
        big = ex._compute_size(0.10)
        assert big > small > 0

    @pytest.mark.asyncio
    async def test_size_capped_at_max(self, na_db):
        cfg = NewsAlphaExecutorConfig(bankroll=1000, max_position_pct=0.05)
        ex = NewsAlphaExecutor(cfg, na_db)
        size = ex._compute_size(0.50)  # Huge edge → should be capped
        assert size <= 1000 * 0.05 + 0.01  # tolerance
