"""Bull researcher agent — argues the YES case."""

from __future__ import annotations

from src.agents.base_agent import BaseAgent, JSON_INSTRUCTION
from src.models.market import UnifiedMarket


class BullResearcherAgent(BaseAgent):
    """
    Bullish case builder — argues for YES outcome.
    Weight: 10% (DeepSeek)
    """

    def build_prompt(self, market: UnifiedMarket, context: dict) -> list[dict]:
        market_info = self._format_market_context(market, context)

        return [
            {
                "role": "system",
                "content": (
                    "You are a research analyst whose role is to build the strongest possible "
                    "case for the YES outcome in prediction markets. You are the team's "
                    "designated bull — your job is to find every reason why YES might happen.\n\n"
                    "Your methodology:\n"
                    "1. Identify the 3-5 strongest arguments supporting YES\n"
                    "2. Find historical precedents where similar situations resulted in YES\n"
                    "3. Identify catalysts that could push the probability higher\n"
                    "4. Consider momentum and trend factors favoring YES\n"
                    "5. Despite your bullish role, still give an honest probability estimate\n\n"
                    "Important: You are an advocate, but still honest. Your probability should "
                    "reflect genuine belief, not just advocacy. Being a bull means you look harder "
                    "for YES evidence, not that you inflate numbers.\n\n"
                    + JSON_INSTRUCTION
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Build the bull case for YES in this market, then give your probability estimate.\n\n"
                    f"{market_info}\n\n"
                    f"What are the strongest arguments for YES? What catalysts or trends support it?"
                ),
            },
        ]
