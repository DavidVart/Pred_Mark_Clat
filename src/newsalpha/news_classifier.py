"""Fast LLM headline classifier using Claude Haiku (or GPT-4o-mini).

Given a news headline + optional summary, returns:
    - sentiment: "bullish" / "bearish" / "neutral"
    - magnitude: 0.0 to 1.0 (how impactful this news is for BTC price)
    - reasoning: one-line explanation

Latency target: <2 seconds end-to-end. We use the smallest capable model
(Haiku 4.5 via OpenRouter) for speed. The classification doesn't need
Opus-level reasoning — it's a simple triage decision.

Cost: ~$0.001 per headline at Haiku pricing. At 20 relevant headlines/hour,
that's $0.02/hour ≈ $0.50/day. Well within budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from src.newsalpha.news_feed import NewsItem
from src.utils.logging import get_logger

logger = get_logger("newsalpha.news_classifier")

# Fast, cheap model for headline triage
DEFAULT_MODEL = "anthropic/claude-haiku-4"

SYSTEM_PROMPT = """You are a crypto trading signal classifier. Given a news headline, classify its likely impact on Bitcoin's price in the next 1-4 hours.

Respond in JSON only:
{"sentiment": "bullish"|"bearish"|"neutral", "magnitude": 0.0-1.0, "reasoning": "<10 words>"}

Magnitude scale:
- 0.0-0.2: routine news, no price impact expected
- 0.2-0.5: meaningful but not market-moving (e.g. minor regulatory statement)
- 0.5-0.8: likely to move BTC 0.5-2% (e.g. ETF flow data, Fed guidance)
- 0.8-1.0: market-moving event (e.g. rate decision surprise, major hack, new ETF approval)

Rules:
- Default to "neutral" with low magnitude when uncertain
- "Crypto company raises funding" = neutral (doesn't move BTC)
- "SEC approves/denies spot BTC ETF" = high magnitude
- "Fed cuts/hikes rates" = high magnitude
- "BTC whale moves $X to exchange" = bearish, medium magnitude
- "OPEC cuts production" = bullish (risk-on), low-medium magnitude
- Ignore clickbait, opinion pieces, price predictions from random analysts"""


@dataclass
class NewsClassification:
    """Result of classifying a news headline."""

    headline: str
    source: str
    sentiment: str  # "bullish" / "bearish" / "neutral"
    magnitude: float  # 0.0 - 1.0
    reasoning: str
    model: str
    latency_ms: int
    cost_usd: float

    @property
    def is_actionable(self) -> bool:
        """True if this headline is worth triggering a market re-evaluation."""
        return self.sentiment != "neutral" and self.magnitude >= 0.3

    @property
    def direction(self) -> str:
        """Maps sentiment to trading direction for BTC-up markets."""
        if self.sentiment == "bullish":
            return "yes"  # bullish → BTC goes up → buy YES
        elif self.sentiment == "bearish":
            return "no"   # bearish → BTC goes down → buy NO
        return "none"


class NewsClassifier:
    """Classifies news headlines via a fast LLM call."""

    def __init__(
        self,
        openrouter_api_key: str,
        model: str = DEFAULT_MODEL,
        http: httpx.AsyncClient | None = None,
    ):
        self.api_key = openrouter_api_key
        self.model = model
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._owns_http = http is None

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def classify(self, item: NewsItem) -> NewsClassification | None:
        """Classify a single headline. Returns None on any failure."""
        user_msg = f"Headline: {item.title}"
        if item.summary:
            user_msg += f"\nSummary: {item.summary[:150]}"

        start = time.monotonic()
        try:
            resp = await self._http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1,  # deterministic classification
                },
            )
            latency_ms = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                logger.warning("classify_http_error", status=resp.status_code, headline=item.title[:50])
                return None

            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Extract cost from OpenRouter response
            usage = data.get("usage", {})
            # Haiku pricing: ~$0.25/M input, $1.25/M output
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost = (prompt_tokens * 0.25 + completion_tokens * 1.25) / 1_000_000

        except Exception as e:
            logger.warning("classify_error", error=str(e), headline=item.title[:50])
            return None

        # Parse JSON from response
        try:
            # Handle markdown code blocks
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("classify_parse_error", content=content[:100], headline=item.title[:50])
            return None

        sentiment = result.get("sentiment", "neutral")
        magnitude = float(result.get("magnitude", 0.0))
        reasoning = result.get("reasoning", "")

        if sentiment not in ("bullish", "bearish", "neutral"):
            sentiment = "neutral"
        magnitude = max(0.0, min(1.0, magnitude))

        classification = NewsClassification(
            headline=item.title,
            source=item.source,
            sentiment=sentiment,
            magnitude=magnitude,
            reasoning=reasoning,
            model=self.model,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

        logger.info(
            "headline_classified",
            headline=item.title[:60],
            sentiment=sentiment,
            magnitude=round(magnitude, 2),
            reasoning=reasoning[:40],
            latency_ms=latency_ms,
            cost=f"${cost:.4f}",
        )

        return classification

    async def classify_batch(self, items: list[NewsItem]) -> list[NewsClassification]:
        """Classify multiple headlines concurrently. Returns only successful ones."""
        import asyncio
        results = await asyncio.gather(
            *[self.classify(item) for item in items],
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, NewsClassification)]
