"""Tests for the ensemble aggregation engine."""

from __future__ import annotations

import pytest

from src.agents.ensemble import EnsembleRunner
from src.models.market import UnifiedMarket
from src.models.prediction import AgentPrediction, EnsembleResult


class MockAgent:
    """Mock agent that returns a fixed prediction."""

    def __init__(self, model_name: str, role: str, weight: float, probability: float, confidence: float):
        self.model_name = model_name
        self.role = role
        self.weight = weight
        self._probability = probability
        self._confidence = confidence

        class _Model:
            def __init__(self, model_id, weight):
                self.model_id = model_id
                self.weight = weight

        self.model = _Model(model_name, weight)

    async def analyze(self, market, context):
        return AgentPrediction(
            model_name=self.model_name,
            role=self.role,
            probability=self._probability,
            confidence=self._confidence,
            reasoning="Mock reasoning",
            cost_usd=0.01,
            tokens_used=100,
        )


class MockFailingAgent(MockAgent):
    async def analyze(self, market, context):
        return AgentPrediction(
            model_name=self.model_name,
            role=self.role,
            probability=0.5,
            confidence=0.0,
            reasoning="",
            error="Mock failure",
        )


@pytest.fixture
def sample_market():
    from datetime import datetime, timedelta
    return UnifiedMarket(
        platform="kalshi",
        market_id="TEST-001",
        title="Test Market",
        yes_price=0.45,
        no_price=0.55,
        volume=10000,
    )


@pytest.fixture
def agreeing_agents():
    """5 agents that roughly agree on 0.65 probability."""
    return [
        MockAgent("model-a", "forecaster", 0.30, 0.65, 0.85),
        MockAgent("model-b", "news_analyst", 0.30, 0.63, 0.80),
        MockAgent("model-c", "risk_manager", 0.20, 0.60, 0.75),
        MockAgent("model-d", "bull_researcher", 0.10, 0.70, 0.70),
        MockAgent("model-e", "bear_researcher", 0.10, 0.62, 0.70),
    ]


@pytest.fixture
def disagreeing_agents():
    """5 agents with high disagreement."""
    return [
        MockAgent("model-a", "forecaster", 0.30, 0.80, 0.85),
        MockAgent("model-b", "news_analyst", 0.30, 0.30, 0.80),
        MockAgent("model-c", "risk_manager", 0.20, 0.50, 0.75),
        MockAgent("model-d", "bull_researcher", 0.10, 0.90, 0.70),
        MockAgent("model-e", "bear_researcher", 0.10, 0.20, 0.70),
    ]


class TestEnsembleAggregation:
    @pytest.mark.asyncio
    async def test_agreeing_ensemble(self, agreeing_agents, sample_market):
        runner = EnsembleRunner(agreeing_agents)
        result = await runner.predict(sample_market, {})

        assert result.models_succeeded == 5
        assert 0.60 <= result.weighted_probability <= 0.70
        assert result.final_confidence > 0.5
        assert result.disagreement_score < 0.1
        assert result.has_consensus

    @pytest.mark.asyncio
    async def test_positive_edge(self, agreeing_agents, sample_market):
        runner = EnsembleRunner(agreeing_agents)
        result = await runner.predict(sample_market, {})

        # Market at 0.45, models predict ~0.65, so edge should be ~+0.20
        assert result.edge > 0.15
        assert result.trade_side == "yes"

    @pytest.mark.asyncio
    async def test_disagreement_penalizes_confidence(self, disagreeing_agents, sample_market):
        runner = EnsembleRunner(disagreeing_agents, disagreement_threshold=0.20)
        result = await runner.predict(sample_market, {})

        assert result.disagreement_score > 0.20
        # Confidence should be penalized due to disagreement
        assert result.final_confidence < 0.80

    @pytest.mark.asyncio
    async def test_insufficient_models(self, sample_market):
        agents = [
            MockAgent("model-a", "forecaster", 0.50, 0.65, 0.85),
            MockFailingAgent("model-b", "news_analyst", 0.30, 0.0, 0.0),
            MockFailingAgent("model-c", "risk_manager", 0.20, 0.0, 0.0),
        ]
        runner = EnsembleRunner(agents, min_models=3)
        result = await runner.predict(sample_market, {})

        # Only 1 model succeeded, need 3
        assert not result.has_consensus
        assert result.final_confidence == 0.0

    @pytest.mark.asyncio
    async def test_partial_failure_still_works(self, sample_market):
        agents = [
            MockAgent("model-a", "forecaster", 0.30, 0.65, 0.85),
            MockAgent("model-b", "news_analyst", 0.30, 0.63, 0.80),
            MockAgent("model-c", "risk_manager", 0.20, 0.60, 0.75),
            MockFailingAgent("model-d", "bull_researcher", 0.10, 0.0, 0.0),
            MockFailingAgent("model-e", "bear_researcher", 0.10, 0.0, 0.0),
        ]
        runner = EnsembleRunner(agents, min_models=3)
        result = await runner.predict(sample_market, {})

        assert result.models_succeeded == 3
        assert result.models_failed == 2
        assert result.has_consensus
        assert result.weighted_probability > 0.5

    @pytest.mark.asyncio
    async def test_cost_tracking(self, agreeing_agents, sample_market):
        runner = EnsembleRunner(agreeing_agents)
        result = await runner.predict(sample_market, {})

        assert result.total_cost_usd > 0
        assert result.total_cost_usd == sum(
            p.cost_usd for p in result.individual_predictions
        )
