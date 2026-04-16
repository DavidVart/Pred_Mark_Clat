"""Track and enforce daily LLM API spend limits."""

from __future__ import annotations

from src.db.manager import DatabaseManager
from src.utils.logging import get_logger

logger = get_logger("cost_tracker")


class CostTracker:
    """Tracks LLM API costs and enforces daily budget."""

    def __init__(self, db: DatabaseManager, daily_limit: float = 10.0):
        self.db = db
        self.daily_limit = daily_limit

    async def can_spend(self, estimated_cost: float = 0.0) -> bool:
        """Check if we're within daily budget."""
        spent = await self.db.get_daily_ai_cost()
        remaining = self.daily_limit - spent
        if remaining <= 0:
            logger.warning("daily_ai_budget_exhausted", spent=spent, limit=self.daily_limit)
            return False
        if estimated_cost > remaining:
            logger.warning(
                "insufficient_ai_budget",
                estimated=estimated_cost,
                remaining=remaining,
            )
            return False
        return True

    async def record_spend(
        self,
        model_name: str,
        role: str,
        market_id: str | None,
        cost_usd: float,
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record an LLM query and its cost."""
        await self.db.log_llm_query({
            "model_name": model_name,
            "role": role,
            "market_id": market_id or "",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": tokens or (prompt_tokens + completion_tokens),
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "success": int(success),
            "error": error,
        })

        if cost_usd > 0:
            logger.debug(
                "llm_cost_recorded",
                model=model_name,
                cost=cost_usd,
                tokens=tokens,
            )

    async def get_daily_spent(self) -> float:
        return await self.db.get_daily_ai_cost()

    async def get_remaining_budget(self) -> float:
        spent = await self.db.get_daily_ai_cost()
        return max(0.0, self.daily_limit - spent)
