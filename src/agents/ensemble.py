"""Ensemble runner — aggregates predictions from multiple agents."""

from __future__ import annotations

import asyncio
import statistics

from config.models import ModelSpec, get_ensemble
from src.agents.base_agent import BaseAgent
from src.agents.bear_researcher import BearResearcherAgent
from src.agents.bull_researcher import BullResearcherAgent
from src.agents.forecaster import ForecasterAgent
from src.agents.news_analyst import NewsAnalystAgent
from src.agents.risk_agent import RiskAgent
from src.clients.openrouter_client import OpenRouterClient
from src.models.market import UnifiedMarket
from src.models.prediction import AgentPrediction, EnsembleResult
from src.utils.logging import get_logger

logger = get_logger("ensemble")

# Map role names to agent classes
AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "forecaster": ForecasterAgent,
    "news_analyst": NewsAnalystAgent,
    "risk_manager": RiskAgent,
    "bull_researcher": BullResearcherAgent,
    "bear_researcher": BearResearcherAgent,
}


def build_agents(client: OpenRouterClient, models: list[ModelSpec] | None = None) -> list[BaseAgent]:
    """Build agent instances from model specs."""
    if models is None:
        models = get_ensemble()

    agents = []
    for spec in models:
        cls = AGENT_CLASSES.get(spec.role)
        if cls is None:
            logger.warning("unknown_agent_role", role=spec.role)
            continue
        agents.append(cls(spec, client))

    return agents


class EnsembleRunner:
    """Runs multiple agents in parallel and aggregates their predictions."""

    def __init__(
        self,
        agents: list[BaseAgent],
        min_models: int = 3,
        disagreement_threshold: float = 0.25,
        disagreement_penalty: float = 0.30,
    ):
        self.agents = agents
        self.min_models = min_models
        self.disagreement_threshold = disagreement_threshold
        self.disagreement_penalty = disagreement_penalty
        self._weights = {a.model_name: a.model.weight for a in agents}

    async def predict(self, market: UnifiedMarket, context: dict) -> EnsembleResult:
        """Run all agents in parallel and aggregate results."""
        logger.info(
            "ensemble_start",
            market=market.title[:60],
            agents=len(self.agents),
        )

        # Run all agents concurrently
        tasks = [agent.analyze(market, context) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes and failures
        predictions: list[AgentPrediction] = []
        failures = 0

        for result in results:
            if isinstance(result, Exception):
                failures += 1
                logger.error("agent_exception", error=str(result))
            elif isinstance(result, AgentPrediction):
                if result.succeeded:
                    predictions.append(result)
                else:
                    failures += 1

        total_cost = sum(p.cost_usd for p in predictions)

        if len(predictions) < self.min_models:
            logger.warning(
                "insufficient_models",
                succeeded=len(predictions),
                required=self.min_models,
            )
            return EnsembleResult(
                market_id=market.market_id,
                weighted_probability=0.5,
                final_confidence=0.0,
                individual_predictions=predictions,
                disagreement_score=1.0,
                edge=0.0,
                total_cost_usd=total_cost,
                models_succeeded=len(predictions),
                models_failed=failures,
            )

        # Aggregate
        ensemble = self._aggregate(predictions, market)
        ensemble.models_failed = failures
        ensemble.total_cost_usd = total_cost

        logger.info(
            "ensemble_complete",
            market=market.title[:60],
            probability=f"{ensemble.weighted_probability:.3f}",
            confidence=f"{ensemble.final_confidence:.3f}",
            edge=f"{ensemble.edge:+.3f}",
            disagreement=f"{ensemble.disagreement_score:.3f}",
            cost=f"${total_cost:.4f}",
            models=f"{len(predictions)}/{len(self.agents)}",
        )

        return ensemble

    def _aggregate(self, predictions: list[AgentPrediction], market: UnifiedMarket) -> EnsembleResult:
        """Confidence-adjusted weighted average of predictions."""
        # Calculate effective weights: base_weight * confidence
        effective_weights = []
        for pred in predictions:
            base_weight = self._weights.get(pred.model_name, 1.0 / len(predictions))
            conf = max(pred.confidence, 0.1)  # Floor at 0.1 to avoid zeroing out
            effective_weights.append(base_weight * conf)

        total_weight = sum(effective_weights)
        if total_weight == 0:
            total_weight = 1.0

        # Weighted average probability
        weighted_prob = sum(
            pred.probability * w for pred, w in zip(predictions, effective_weights)
        ) / total_weight

        # Weighted average confidence
        weighted_conf = sum(
            pred.confidence * w for pred, w in zip(predictions, effective_weights)
        ) / total_weight

        # Disagreement = standard deviation of probabilities
        probabilities = [p.probability for p in predictions]
        disagreement = statistics.stdev(probabilities) if len(probabilities) > 1 else 0.0

        # Penalize confidence if models disagree significantly
        if disagreement > self.disagreement_threshold:
            penalty = min(self.disagreement_penalty, disagreement)
            weighted_conf *= (1.0 - penalty)
            logger.info(
                "disagreement_penalty",
                disagreement=f"{disagreement:.3f}",
                penalty=f"{penalty:.3f}",
            )

        # Calculate edge vs market
        edge = weighted_prob - market.yes_price

        return EnsembleResult(
            market_id=market.market_id,
            weighted_probability=max(0.0, min(1.0, weighted_prob)),
            final_confidence=max(0.0, min(1.0, weighted_conf)),
            individual_predictions=predictions,
            disagreement_score=disagreement,
            edge=edge,
            models_succeeded=len(predictions),
        )
