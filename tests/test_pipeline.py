"""Tests for the database manager and pipeline components."""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.db.manager import DatabaseManager


class TestDatabaseManager:
    @pytest.mark.asyncio
    async def test_initialize(self, db):
        """DB should initialize without errors."""
        # Kill switch should have default row
        ks = await db.get_kill_switch()
        assert ks["active"] == 0

    @pytest.mark.asyncio
    async def test_upsert_market(self, db):
        market = {
            "market_id": "test-123",
            "platform": "kalshi",
            "title": "Test Market",
            "description": "A test",
            "category": "weather",
            "yes_price": 0.45,
            "no_price": 0.55,
            "volume": 10000,
            "liquidity": 5000.0,
            "expiration": "2026-05-01T00:00:00",
            "status": "active",
            "outcomes": ["Yes", "No"],
            "url": "https://kalshi.com/test",
            "ticker": "TEST-123",
            "clob_token_ids": None,
            "condition_id": None,
        }
        await db.upsert_market(market)

        fetched = await db.get_market("test-123")
        assert fetched is not None
        assert fetched["title"] == "Test Market"
        assert fetched["yes_price"] == 0.45

    @pytest.mark.asyncio
    async def test_position_lifecycle(self, db):
        """Test insert, update, close position flow."""
        # Insert parent market first (FK constraint)
        await db.upsert_market({
            "market_id": "mkt-001", "platform": "kalshi", "title": "Test",
            "description": "", "category": "", "yes_price": 0.5, "no_price": 0.5,
            "volume": 1000, "liquidity": 0, "expiration": None, "status": "active",
            "outcomes": ["Yes", "No"], "url": "", "ticker": "TST", "clob_token_ids": None,
            "condition_id": None,
        })
        pos = {
            "position_id": "pos-001",
            "market_id": "mkt-001",
            "platform": "kalshi",
            "title": "Test Position",
            "side": "yes",
            "entry_price": 0.50,
            "quantity": 10.0,
            "cost_basis": 50.0,
            "current_price": 0.50,
            "stop_loss": 0.10,
            "take_profit": 0.25,
            "is_paper": 1,
            "category": "weather",
            "opened_at": "2026-04-13T10:00:00",
        }
        await db.insert_position(pos)

        positions = await db.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["position_id"] == "pos-001"

        # Update price
        await db.update_position_price("pos-001", 0.60)
        positions = await db.get_open_positions()
        assert positions[0]["current_price"] == 0.60

        # Close position
        await db.close_position("pos-001", exit_price=0.65, pnl=1.50, pnl_pct=0.03, outcome="win")
        positions = await db.get_open_positions()
        assert len(positions) == 0

        # Check trade log
        trades = await db.get_trade_history()
        assert len(trades) == 1
        assert trades[0]["pnl"] == 1.50
        assert trades[0]["outcome"] == "win"

    @pytest.mark.asyncio
    async def test_kill_switch(self, db):
        assert not await db.is_kill_switch_active()

        await db.set_kill_switch(True, "Test reason", "test")
        assert await db.is_kill_switch_active()

        ks = await db.get_kill_switch()
        assert ks["reason"] == "Test reason"

        await db.set_kill_switch(False)
        assert not await db.is_kill_switch_active()

    @pytest.mark.asyncio
    async def test_cooldown(self, db):
        assert not await db.is_on_cooldown("mkt-001")

        await db.record_analysis("mkt-001")
        assert await db.is_on_cooldown("mkt-001", cooldown_hours=1.0)

        count = await db.get_daily_analysis_count("mkt-001")
        assert count == 1

    @pytest.mark.asyncio
    async def test_llm_cost_tracking(self, db):
        await db.log_llm_query({
            "model_name": "test-model",
            "role": "forecaster",
            "market_id": "mkt-001",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.05,
            "duration_ms": 500,
            "success": 1,
            "error": None,
        })

        cost = await db.get_daily_ai_cost()
        assert cost == 0.05

    @pytest.mark.asyncio
    async def test_win_rate_empty(self, db):
        rate = await db.get_win_rate()
        assert rate == 0.0
