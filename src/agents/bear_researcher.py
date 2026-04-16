"""Bear researcher agent — argues the NO case."""

from __future__ import annotations

from src.agents.base_agent import BaseAgent, JSON_INSTRUCTION
from src.models.market import UnifiedMarket


class BearResearcherAgent(BaseAgent):
    """
    Bearish case builder — argues for NO outcome.
    Weight: 10% (Grok)
    """

    def build_prompt(self, market: UnifiedMarket, context: dict) -> list[dict]:
        market_info = self._format_market_context(market, context)

        return [
            {
                "role": "system",
                "content": (
                    "You are a research analyst whose role is to build the strongest possible "
                    "case for the NO outcome in prediction markets. You are the team's "
                    "designated bear — your job is to find every reason why YES might NOT happen.\n\n"
                    "Your methodology:\n"
                    "1. Identify the 3-5 strongest arguments against YES (supporting NO)\n"
                    "2. Find historical precedents where similar situations resulted in NO\n"
                    "3. Identify risks and obstacles that could prevent YES from happening\n"
                    "4. Consider mean-reversion and overreaction factors\n"
                    "5. Despite your bearish role, still give an honest probability estimate\n\n"
                    "Important: You are a skeptic, but still honest. Your probability should "
                    "reflect genuine belief. Being a bear means you look harder for NO evidence, "
                    "not that you deflate numbers blindly.\n\n"
                    + JSON_INSTRUCTION
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Build the bear case against YES in this market, then give your probability estimate.\n\n"
                    f"{market_info}\n\n"
                    f"What are the strongest arguments for NO? What obstacles or risks threaten YES?"
                ),
            },
        ]
