"""Abstract base agent for LLM-powered market analysis."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod

from config.models import ModelSpec
from src.clients.openrouter_client import OpenRouterClient
from src.models.market import UnifiedMarket
from src.models.prediction import AgentPrediction
from src.utils.logging import get_logger

logger = get_logger("agent")

# Shared system instruction for JSON output
JSON_INSTRUCTION = """
You MUST respond with valid JSON only. No markdown, no explanation outside the JSON.
Format:
{
    "probability": <float 0.0-1.0>,
    "confidence": <float 0.0-1.0>,
    "reasoning": "<brief explanation>"
}
"""


class BaseAgent(ABC):
    """Abstract base class for ensemble agents."""

    def __init__(self, model_spec: ModelSpec, client: OpenRouterClient):
        self.model = model_spec
        self.client = client

    @property
    def role(self) -> str:
        return self.model.role

    @property
    def model_name(self) -> str:
        return self.model.model_id

    @abstractmethod
    def build_prompt(self, market: UnifiedMarket, context: dict) -> list[dict]:
        """Build the chat messages for this agent's analysis."""
        ...

    async def analyze(self, market: UnifiedMarket, context: dict) -> AgentPrediction:
        """Run analysis on a market. Returns AgentPrediction."""
        start = time.monotonic()
        try:
            messages = self.build_prompt(market, context)
            text, cost, tokens = await self.client.complete(
                model=self.model.model_id,
                messages=messages,
                max_tokens=self.model.max_tokens,
                temperature=0.3,
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            prediction = self._parse_response(text)
            prediction.model_name = self.model.model_id
            prediction.role = self.role
            prediction.cost_usd = cost
            prediction.tokens_used = tokens

            logger.debug(
                "agent_prediction",
                role=self.role,
                model=self.model.model_id,
                probability=prediction.probability,
                confidence=prediction.confidence,
                cost=f"${cost:.4f}",
                duration_ms=duration_ms,
            )

            return prediction

        except Exception as e:
            logger.error("agent_failed", role=self.role, model=self.model.model_id, error=str(e))
            return AgentPrediction(
                model_name=self.model.model_id,
                role=self.role,
                probability=0.5,
                confidence=0.0,
                reasoning="",
                error=str(e),
            )

    def _parse_response(self, text: str) -> AgentPrediction:
        """Parse JSON response from the LLM."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]*"probability"[^{}]*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract numbers
            prob_match = re.search(r'"probability"\s*:\s*([\d.]+)', text)
            conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)

            probability = float(prob_match.group(1)) if prob_match else 0.5
            confidence = float(conf_match.group(1)) if conf_match else 0.0
            reasoning = reason_match.group(1) if reason_match else text[:200]

            return AgentPrediction(
                model_name="",
                role="",
                probability=max(0.0, min(1.0, probability)),
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=reasoning,
            )

        probability = float(data.get("probability", 0.5))
        confidence = float(data.get("confidence", 0.0))
        reasoning = str(data.get("reasoning", ""))

        return AgentPrediction(
            model_name="",
            role="",
            probability=max(0.0, min(1.0, probability)),
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning,
        )

    def _format_market_context(self, market: UnifiedMarket, context: dict) -> str:
        """Format market data for the prompt."""
        parts = [
            f"Market: {market.title}",
            f"Description: {market.description}" if market.description else "",
            f"Platform: {market.platform}",
            f"Current YES price: {market.yes_price:.2f} ({market.yes_price:.0%})",
            f"Current NO price: {market.no_price:.2f} ({market.no_price:.0%})",
            f"Volume: {market.volume:,}",
            f"Category: {market.category}" if market.category else "",
        ]

        if market.time_to_expiry_hours is not None:
            hours = market.time_to_expiry_hours
            if hours > 24:
                parts.append(f"Time to expiry: {hours / 24:.1f} days")
            else:
                parts.append(f"Time to expiry: {hours:.1f} hours")

        # Add research context if available
        news = context.get("news_headlines", [])
        if news:
            parts.append(f"\nRecent news ({len(news)} items):")
            for headline in news[:10]:
                parts.append(f"  - {headline}")

        sentiment = context.get("sentiment_summary")
        if sentiment:
            parts.append(f"\nSentiment summary: {sentiment}")

        return "\n".join(p for p in parts if p)
