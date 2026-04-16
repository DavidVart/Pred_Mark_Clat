"""Tests for the deterministic risk manager."""

from __future__ import annotations

import pytest
import pytest_asyncio

from config.settings import RiskConfig
from src.db.manager import DatabaseManager
from src.models.market import UnifiedMarket
from src.models.portfolio import PortfolioState
from src.models.prediction import EnsembleResult
from src.models.trade import Position
from src.pipeline.risk_manager import RiskManager


@pytest_asyncio.fixture
async def risk_mgr(db):
    return RiskManager(RiskConfig(), db)


class TestKellyCriterion:
    def test_positive_edge(self, risk_mgr):
        """Positive edge should produce positive Kelly fraction."""
        size = risk_mgr._kelly_criterion(win_prob=0.65, price=0.50, confidence=0.80)
        assert size > 0
        assert size <= risk_mgr.config.max_position_size_pct

    def test_no_edge(self, risk_mgr):
        """No edge should produce zero or near-zero sizing."""
        size = risk_mgr._kelly_criterion(win_prob=0.50, price=0.50, confidence=0.80)
        assert size == 0.0

    def test_negative_edge(self, risk_mgr):
        """Negative edge should produce zero sizing."""
        size = risk_mgr._kelly_criterion(win_prob=0.30, price=0.50, confidence=0.80)
        assert size == 0.0

    def test_capped_at_max(self, risk_mgr):
        """Even extreme edge should cap at max position size."""
        size = risk_mgr._kelly_criterion(win_prob=0.99, price=0.10, confidence=1.0)
        assert size <= risk_mgr.config.max_position_size_pct

    def test_confidence_scaling(self, risk_mgr):
        """Lower confidence should reduce position size."""
        high_conf = risk_mgr._kelly_criterion(win_prob=0.55, price=0.50, confidence=0.90)
        low_conf = risk_mgr._kelly_criterion(win_prob=0.55, price=0.50, confidence=0.50)
        assert high_conf > low_conf

    def test_zero_price(self, risk_mgr):
        """Zero price should return zero."""
        size = risk_mgr._kelly_criterion(win_prob=0.70, price=0.0, confidence=0.80)
        assert size == 0.0


class TestRiskChecks:
    @pytest.mark.asyncio
    async def test_approve_valid_trade(self, risk_mgr, sample_market):
        """Valid trade should be approved."""
        prediction = EnsembleResult(
            market_id=sample_market.market_id,
            weighted_probability=0.65,
            final_confidence=0.80,
            edge=0.20,
            models_succeeded=5,
        )
        portfolio = PortfolioState(total_value=1000.0, peak_value=1000.0)

        signal, reason = await risk_mgr.approve_trade(sample_market, prediction, portfolio)
        assert signal is not None
        assert reason == ""
        assert signal.dollar_size > 0

    @pytest.mark.asyncio
    async def test_reject_low_edge(self, risk_mgr, sample_market):
        """Trade with edge below minimum should be rejected."""
        prediction = EnsembleResult(
            market_id=sample_market.market_id,
            weighted_probability=0.47,
            final_confidence=0.80,
            edge=0.02,
            models_succeeded=5,
        )
        portfolio = PortfolioState(total_value=1000.0, peak_value=1000.0)

        signal, reason = await risk_mgr.approve_trade(sample_market, prediction, portfolio)
        assert signal is None
        assert "Edge" in reason

    @pytest.mark.asyncio
    async def test_reject_low_confidence(self, risk_mgr, sample_market):
        """Trade with confidence below minimum should be rejected."""
        prediction = EnsembleResult(
            market_id=sample_market.market_id,
            weighted_probability=0.70,
            final_confidence=0.30,
            edge=0.25,
            models_succeeded=5,
        )
        portfolio = PortfolioState(total_value=1000.0, peak_value=1000.0)

        signal, reason = await risk_mgr.approve_trade(sample_market, prediction, portfolio)
        assert signal is None
        assert "Confidence" in reason

    @pytest.mark.asyncio
    async def test_reject_max_positions(self, risk_mgr, sample_market):
        """Should reject when position limit reached."""
        positions = [
            Position(
                position_id=f"pos-{i}",
                market_id=f"mkt-{i}",
                platform="kalshi",
                title=f"Market {i}",
                side="yes",
                entry_price=0.50,
                quantity=10,
                cost_basis=50.0,
            )
            for i in range(10)
        ]
        prediction = EnsembleResult(
            market_id=sample_market.market_id,
            weighted_probability=0.70,
            final_confidence=0.80,
            edge=0.25,
            models_succeeded=5,
        )
        portfolio = PortfolioState(
            total_value=1000.0, peak_value=1000.0, positions=positions
        )

        signal, reason = await risk_mgr.approve_trade(sample_market, prediction, portfolio)
        assert signal is None
        assert "Max positions" in reason

    @pytest.mark.asyncio
    async def test_reject_drawdown_breach(self, risk_mgr, sample_market):
        """Should reject and activate kill switch on drawdown breach."""
        prediction = EnsembleResult(
            market_id=sample_market.market_id,
            weighted_probability=0.70,
            final_confidence=0.80,
            edge=0.25,
            models_succeeded=5,
        )
        portfolio = PortfolioState(
            total_value=800.0,  # 20% drawdown from peak
            peak_value=1000.0,
        )

        signal, reason = await risk_mgr.approve_trade(sample_market, prediction, portfolio)
        assert signal is None
        assert "drawdown" in reason.lower()


class TestVaR:
    def test_empty_portfolio(self, risk_mgr):
        portfolio = PortfolioState(total_value=1000.0)
        var = risk_mgr.calculate_var_95(portfolio)
        assert var == 0.0

    def test_with_positions(self, risk_mgr):
        positions = [
            Position(
                position_id="p1", market_id="m1", platform="kalshi",
                title="Test", side="yes", entry_price=0.5,
                quantity=10, cost_basis=100.0,
            )
        ]
        portfolio = PortfolioState(total_value=1000.0, positions=positions)
        var = risk_mgr.calculate_var_95(portfolio)
        assert var > 0
