"""RSS-based news feed poller for crypto/macro headlines.

Polls a curated list of free RSS feeds every N seconds, deduplicates by title
hash, and emits new headlines for classification. No API keys needed — these
are all public RSS feeds.

Feed selection rationale:
    - CoinDesk: primary crypto news wire, fast on BTC-moving events
    - The Block: institutional crypto, ETF flows, regulatory
    - Reuters Business: macro, Fed, oil, geopolitical shocks
    - CoinTelegraph: high-volume crypto headlines (noisy but fast)
    - Google News (BTC topic): aggregator catch-all

We intentionally keep the feed list short. Each additional feed adds latency
(serial HTTP fetches) and noise. Quality > quantity.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime

import feedparser
import httpx

from src.utils.logging import get_logger

logger = get_logger("newsalpha.news_feed")


# Curated free RSS feeds. Ordered by typical latency (fastest first).
DEFAULT_FEEDS: list[dict[str, str]] = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "The Block", "url": "https://www.theblock.co/rss.xml"},
    {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"name": "Google News BTC", "url": "https://news.google.com/rss/search?q=bitcoin+OR+btc+OR+crypto&hl=en-US&gl=US&ceid=US:en"},
]

# Keywords that make a headline relevant to our BTC trading.
# Case-insensitive matching. If NONE of these appear, skip the headline.
BTC_KEYWORDS = [
    "bitcoin", "btc", "crypto", "ethereum", "eth",
    "fed ", "federal reserve", "fomc", "rate cut", "rate hike",
    "sec ", "etf", "spot etf",
    "binance", "coinbase", "kraken",
    "stablecoin", "tether", "usdt", "usdc",
    "liquidation", "whale", "halving",
    "inflation", "cpi", "ppi", "nonfarm", "jobs report",
    "treasury", "bond yield", "dollar index", "dxy",
    "oil price", "opec", "crude",
]


@dataclass
class NewsItem:
    """A single news headline with metadata."""

    title: str
    source: str
    url: str
    published: datetime | None = None
    summary: str = ""
    content_hash: str = ""  # for dedup

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                (self.title + self.url).encode()
            ).hexdigest()[:16]


class NewsFeed:
    """Polls RSS feeds and emits new, relevant headlines."""

    def __init__(
        self,
        feeds: list[dict[str, str]] | None = None,
        keywords: list[str] | None = None,
        max_age_seconds: float = 3600.0,  # ignore headlines older than 1h
        http: httpx.AsyncClient | None = None,
    ):
        self.feeds = feeds or DEFAULT_FEEDS
        self.keywords = [k.lower() for k in (keywords or BTC_KEYWORDS)]
        self.max_age_seconds = max_age_seconds
        self._http = http or httpx.AsyncClient(timeout=8.0)
        self._owns_http = http is None
        # Dedup: set of content_hash values we've already emitted
        self._seen: set[str] = set()
        # Cap the seen-set at 10k to prevent memory leak on long runs
        self._seen_max = 10_000

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def poll(self) -> list[NewsItem]:
        """Fetch all feeds, filter to relevant new headlines, return them.

        Safe to call in a loop every 30-60 seconds. Deduplicates across calls.
        """
        all_items: list[NewsItem] = []

        for feed_config in self.feeds:
            try:
                items = await self._fetch_feed(feed_config)
                all_items.extend(items)
            except Exception as e:
                logger.debug("feed_error", feed=feed_config["name"], error=str(e))

        # Filter: relevant keywords + not seen before
        new_items: list[NewsItem] = []
        for item in all_items:
            if item.content_hash in self._seen:
                continue
            if not self._is_relevant(item):
                continue
            self._seen.add(item.content_hash)
            new_items.append(item)

        # Trim seen-set if it's gotten too large
        if len(self._seen) > self._seen_max:
            # Keep the most recent half
            self._seen = set(list(self._seen)[self._seen_max // 2:])

        if new_items:
            logger.info("new_headlines", count=len(new_items),
                        sources=[i.source for i in new_items[:5]])

        return new_items

    async def _fetch_feed(self, feed_config: dict) -> list[NewsItem]:
        """Fetch and parse a single RSS feed."""
        name = feed_config["name"]
        url = feed_config["url"]

        try:
            resp = await self._http.get(url)
            if resp.status_code != 200:
                return []
            raw = resp.text
        except Exception:
            return []

        # feedparser is sync but fast on small payloads
        parsed = feedparser.parse(raw)
        items: list[NewsItem] = []
        now = time.time()

        for entry in parsed.entries[:20]:  # cap per-feed to avoid noise
            title = entry.get("title", "").strip()
            if not title:
                continue
            link = entry.get("link", "")
            summary = entry.get("summary", "")[:200]

            # Parse published date if available
            published = None
            for date_field in ("published_parsed", "updated_parsed"):
                tp = entry.get(date_field)
                if tp:
                    try:
                        import calendar
                        epoch = calendar.timegm(tp)
                        if now - epoch > self.max_age_seconds:
                            continue  # too old
                        published = datetime.utcfromtimestamp(epoch)
                    except Exception:
                        pass
                    break

            items.append(NewsItem(
                title=title,
                source=name,
                url=link,
                published=published,
                summary=summary,
            ))

        return items

    def _is_relevant(self, item: NewsItem) -> bool:
        """Check if a headline matches our keyword filter."""
        text = (item.title + " " + item.summary).lower()
        return any(kw in text for kw in self.keywords)
