"""Risk manager agent — identifies tail risks and sizing recommendations."""

from __future__ import annotations

from src.agents.base_agent import BaseAgent, JSON_INSTRUCTION
from src.models.market import UnifiedMarket


class RiskAgent(BaseAgent):
    """
    Risk assessment and tail-event identification.
    Weight: 20% (GPT-4.1)
    """

    def build_prompt(self, market: UnifiedMarket, context: dict) -> list[dict]:
        market_info = self._format_market_context(market, context)

        return [
            {
                "role": "system",
                "content": (
                    "You are a quantitative risk manager for a prediction market fund. "
                    "Your primary concern is identifying risks that could cause losses.\n\n"
                    "Your methodology:\n"
                    "1. Identify the top 3 risks that could make this trade lose money\n"
                    "2. Assess tail risks — low-probability, high-impact scenarios\n"
                    "3. Consider liquidity risk, timing risk, and information risk\n"
                    "4. Evaluate the confidence level appropriate for this market\n"
                    "5. Estimate the true probability conservatively — when in doubt, shade toward 50%\n\n"
                    "You tend to be more cautious than other analysts. Your role is to prevent "
                    "overconfident bets and flag hidden dangers.\n\n"
                    + JSON_INSTRUCTION
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Assess the risk and estimate the probability for this market.\n\n"
                    f"{market_info}\n\n"
                    f"Focus on: What could go wrong? What are the tail risks? How much "
                    f"uncertainty is there? If the market is pricing YES at {market.yes_price:.0%}, "
                    f"are there hidden risks that justify a different probability?"
                ),
            },
        ]
