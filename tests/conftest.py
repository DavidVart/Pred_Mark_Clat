"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from config.settings import RiskConfig, ScannerConfig, CostConfig, EnsembleConfig
from src.db.manager import DatabaseManager
from src.models.market import UnifiedMarket
from src.models.prediction import AgentPrediction, EnsembleResult


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db():
    """In-memory database for testing."""
    manager = DatabaseManager(":memory:")
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def sample_market() -> UnifiedMarket:
    return UnifiedMarket(
        platform="kalshi",
        market_id="TEST-MARKET-001",
        title="Will it rain in NYC tomorrow?",
        description="Resolves YES if measurable precipitation occurs in Manhattan.",
        category="weather",
        yes_price=0.45,
        no_price=0.55,
        volume=50_000,
        expiration=datetime.utcnow() + timedelta(days=2),
        status="active",
        ticker="RAIN-NYC-2026",
    )


@pytest.fixture
def sample_prediction() -> AgentPrediction:
    return AgentPrediction(
        model_name="test-model",
        role="forecaster",
        probability=0.65,
        confidence=0.80,
        reasoning="Strong weather signals indicate rain.",
        cost_usd=0.01,
        tokens_used=500,
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig()


@pytest.fixture
def scanner_config() -> ScannerConfig:
    return ScannerConfig(min_volume=100)


@pytest.fixture
def cost_config() -> CostConfig:
    return CostConfig()


@pytest.fixture
def ensemble_config() -> EnsembleConfig:
    return EnsembleConfig()
