"""Prediction and ensemble result models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AgentPrediction(BaseModel):
    """Output from a single LLM agent."""

    model_name: str
    role: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    cost_usd: float = 0.0
    tokens_used: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class EnsembleResult(BaseModel):
    """Aggregated output from the ensemble of agents."""

    market_id: str
    weighted_probability: float = Field(ge=0.0, le=1.0)
    final_confidence: float = Field(ge=0.0, le=1.0)
    individual_predictions: list[AgentPrediction] = Field(default_factory=list)
    disagreement_score: float = 0.0
    edge: float = 0.0  # weighted_probability - market_price
    total_cost_usd: float = 0.0
    models_succeeded: int = 0
    models_failed: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def has_consensus(self) -> bool:
        return self.models_succeeded >= 3

    @property
    def trade_side(self) -> str:
        """Which side to trade based on edge direction."""
        return "yes" if self.edge > 0 else "no"

    @property
    def abs_edge(self) -> float:
        return abs(self.edge)
