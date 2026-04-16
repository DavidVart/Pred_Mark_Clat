"""Stage 2: Market Researcher — gather context for each market (no LLM calls)."""

from __future__ import annotations

import asyncio
import re

import feedparser
import httpx

from src.models.market import UnifiedMarket
from src.utils.logging import get_logger

logger = get_logger("researcher")

# RSS feeds to check for relevant news
NEWS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://news.google.com/rss",
]


class MarketResearcher:
    """Gathers context for markets — news, orderbook data, etc. No LLM calls."""

    def __init__(self):
        self._http = httpx.AsyncClient(timeout=15.0)

    async def research(self, market: UnifiedMarket) -> dict:
        """Gather research context for a single market."""
        context: dict = {
            "news_headlines": [],
            "sentiment_summary": "",
            "orderbook_spread": None,
            "volume_rank": None,
        }

        # Extract keywords from market title
        keywords = self._extract_keywords(market.title)

        if keywords:
            headlines = await self._fetch_relevant_news(keywords)
            context["news_headlines"] = headlines[:15]

            if headlines:
                # Simple sentiment heuristic (no LLM needed)
                context["sentiment_summary"] = self._basic_sentiment(headlines)

        logger.debug(
            "research_complete",
            market=market.title[:50],
            headlines=len(context["news_headlines"]),
        )

        return context

    async def research_batch(
        self, markets: list[UnifiedMarket]
    ) -> list[tuple[UnifiedMarket, dict]]:
        """Research multiple markets concurrently."""
        tasks = [self.research(market) for market in markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        enriched = []
        for market, result in zip(markets, results):
            if isinstance(result, Exception):
                logger.error("research_failed", market=market.title[:50], error=str(result))
                enriched.append((market, {}))
            else:
                enriched.append((market, result))

        return enriched

    async def _fetch_relevant_news(self, keywords: list[str]) -> list[str]:
        """Fetch news headlines matching keywords from RSS feeds."""
        headlines: list[str] = []

        async def fetch_feed(url: str) -> list[str]:
            try:
                resp = await self._http.get(url)
                feed = feedparser.parse(resp.text)
                return [entry.get("title", "") for entry in feed.entries[:30]]
            except Exception:
                return []

        # Fetch all feeds concurrently
        tasks = [fetch_feed(url) for url in NEWS_FEEDS]
        results = await asyncio.gather(*tasks)

        all_headlines = []
        for feed_headlines in results:
            all_headlines.extend(feed_headlines)

        # Filter headlines that match any keyword
        keyword_lower = [k.lower() for k in keywords]
        for headline in all_headlines:
            headline_lower = headline.lower()
            if any(kw in headline_lower for kw in keyword_lower):
                headlines.append(headline)

        return headlines

    def _extract_keywords(self, title: str) -> list[str]:
        """Extract meaningful keywords from a market title."""
        # Remove common prediction market boilerplate
        stop_phrases = [
            "will", "won't", "be", "by", "before", "after", "the", "a", "an",
            "to", "in", "on", "at", "of", "for", "is", "are", "was", "were",
            "yes", "no", "market", "prediction", "outcome",
        ]

        words = re.findall(r'\b[A-Za-z]{3,}\b', title)
        keywords = [w for w in words if w.lower() not in stop_phrases]

        # Also try multi-word phrases (names, entities)
        # Keep capitalized words as potential proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
        keywords.extend(proper_nouns)

        # Deduplicate, keep order
        seen = set()
        unique = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique.append(kw)

        return unique[:8]

    def _basic_sentiment(self, headlines: list[str]) -> str:
        """Simple keyword-based sentiment without LLM."""
        positive_words = {"win", "rise", "surge", "gain", "pass", "approve", "success", "agreement", "deal", "boost"}
        negative_words = {"fail", "fall", "drop", "decline", "reject", "crisis", "crash", "loss", "concern", "risk"}

        pos_count = 0
        neg_count = 0

        for h in headlines:
            h_lower = h.lower()
            pos_count += sum(1 for w in positive_words if w in h_lower)
            neg_count += sum(1 for w in negative_words if w in h_lower)

        total = pos_count + neg_count
        if total == 0:
            return "neutral"
        ratio = pos_count / total
        if ratio > 0.6:
            return "bullish"
        elif ratio < 0.4:
            return "bearish"
        return "mixed"

    async def close(self) -> None:
        await self._http.aclose()
