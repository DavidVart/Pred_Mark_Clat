"""LLM model registry for the ensemble."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str        # OpenRouter model ID
    role: str            # Agent role
    weight: float        # Ensemble weight (0-1, sum to 1.0)
    max_tokens: int = 2048
    description: str = ""


# Default 5-model ensemble configuration
DEFAULT_ENSEMBLE: list[ModelSpec] = [
    ModelSpec(
        model_id="anthropic/claude-sonnet-4",
        role="forecaster",
        weight=0.30,
        description="Lead superforecaster — probability estimation specialist",
    ),
    ModelSpec(
        model_id="google/gemini-2.5-pro",
        role="news_analyst",
        weight=0.30,
        description="News and sentiment analysis specialist",
    ),
    ModelSpec(
        model_id="openai/gpt-4.1",
        role="risk_manager",
        weight=0.20,
        description="Risk assessment and tail-event identification",
    ),
    ModelSpec(
        model_id="deepseek/deepseek-r1",
        role="bull_researcher",
        weight=0.10,
        description="Bullish case builder — argues for YES",
    ),
    ModelSpec(
        model_id="x-ai/grok-3",
        role="bear_researcher",
        weight=0.10,
        description="Bearish case builder — argues for NO",
    ),
]


def get_ensemble() -> list[ModelSpec]:
    """Return the default ensemble configuration."""
    return list(DEFAULT_ENSEMBLE)
