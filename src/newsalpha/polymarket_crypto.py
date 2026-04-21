"""Discover and poll Polymarket short-duration BTC markets.

We use TWO APIs:
    - Gamma (https://gamma-api.polymarket.com/markets) for discovery + metadata.
      Returns market titles, CLOB token IDs, start/end dates. BUT the
      `outcomePrices` field here is stale — don't use it for trading.
    - CLOB (https://clob.polymarket.com/midpoint) for LIVE prices. Queried
      per-market with the CLOB token IDs from the Gamma response.

Markets of interest:
    - "Bitcoin Up or Down - <date>, <time>-<time> ET" (5M, 15M, 1h, 4h variants)
    - Any Bitcoin close-price markets with YES/NO structure

We fetch all active markets ending in the next 24h, filter by title pattern,
then query CLOB for each match. Returns one MarketQuote per market.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone

import httpx

from src.newsalpha.coinbase_rest import historical_price
from src.newsalpha.market_classifier import classify_title
from src.newsalpha.models import MarketQuote
from src.utils.logging import get_logger

logger = get_logger("newsalpha.polymarket")

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_MIDPOINT_URL = "https://clob.polymarket.com/midpoint"

# Broad filter at discovery — we still need the classifier to verify supported type.
TITLE_PATTERNS = [
    re.compile(r"bitcoin.*up or down", re.IGNORECASE),
    re.compile(r"btc.*up or down", re.IGNORECASE),
    re.compile(r"bitcoin.*(higher|lower|above|below)", re.IGNORECASE),
    re.compile(r"btc.*(higher|lower|above|below)", re.IGNORECASE),
]

# Max time to resolution we consider. Wider than the signal's own max to let
# the detector see longer-duration markets and decide.
DEFAULT_MAX_HOURS = 24.0


class PolymarketCryptoFeed:
    """Finds Polymarket BTC short-duration markets and quotes them via CLOB."""

    def __init__(
        self,
        http: httpx.AsyncClient | None = None,
        max_hours_to_resolution: float = DEFAULT_MAX_HOURS,
    ):
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._owns_http = http is None
        self.max_hours_to_resolution = max_hours_to_resolution
        # In-memory cache of the ref price we snapshot the first time we see
        # a market. Polymarket doesn't expose the opening strike reliably, so
        # we record "spot at first sighting" and use that going forward.
        # Keyed by market_id → (ref_price, timestamp).
        self._ref_price_cache: dict[str, tuple[float, datetime]] = {}

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def record_ref_price(self, market_id: str, price: float) -> None:
        """Set/overwrite the reference price for a market (strike).

        Called by the orchestrator when a new market is first seen, so we
        have a sensible strike to evaluate against later.
        """
        if market_id not in self._ref_price_cache:
            self._ref_price_cache[market_id] = (price, datetime.utcnow())
            logger.info("ref_price_recorded", market=market_id[:30], price=round(price, 2))

    def get_ref_price(self, market_id: str) -> float | None:
        entry = self._ref_price_cache.get(market_id)
        return entry[0] if entry else None

    async def fetch_active_btc_markets(self, current_spot: float | None = None) -> list[MarketQuote]:
        """Return active Polymarket BTC short-duration markets with LIVE CLOB prices.

        If current_spot is provided, unknown ref_prices for new markets are
        snapshotted to that value on first sight.
        """
        # Discovery: two passes (asc + desc on endDate) to catch both imminent
        # and farther-out markets in the volume-limited Gamma response.
        candidates = await self._discover_candidates()
        logger.info("poly_crypto_candidates", count=len(candidates))

        # Live price: hit CLOB for each candidate's YES/NO token.
        # We do these concurrently but bounded.
        sem = asyncio.Semaphore(10)
        async def _with_prices(m):
            async with sem:
                return await self._build_quote(m, current_spot)

        results = await asyncio.gather(*[_with_prices(m) for m in candidates])
        quotes = [q for q in results if q is not None]
        logger.info("poly_crypto_fetched", total=len(candidates), matched=len(quotes))
        return quotes

    async def _discover_candidates(self) -> list[dict]:
        """Hit Gamma with a couple of orderings to fish out BTC markets in the window."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=self.max_hours_to_resolution)
        seen_ids: set = set()
        out: list[dict] = []

        for order, asc in [("endDate", "true"), ("endDate", "false"), ("volume24hr", "false")]:
            params = {
                "active": "true",
                "closed": "false",
                "limit": "500",
                "order": order,
                "ascending": asc,
            }
            try:
                resp = await self._http.get(GAMMA_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning("gamma_fetch_error", error=str(e))
                continue

            markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
            for m in markets:
                if m.get("id") in seen_ids:
                    continue
                title = m.get("question") or m.get("title") or ""
                if not any(p.search(title) for p in TITLE_PATTERNS):
                    continue

                end_iso = m.get("endDate") or m.get("endTime")
                if not end_iso:
                    continue
                try:
                    end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if end <= now or end > cutoff:
                    continue

                seen_ids.add(m.get("id"))
                out.append(m)

        return out

    async def _build_quote(self, m: dict, current_spot: float | None) -> MarketQuote | None:
        title = m.get("question") or m.get("title") or ""
        market_id = m.get("conditionId") or m.get("id") or title[:60]

        # CRITICAL: classify the market BEFORE we compute strike. Using the wrong
        # strike produces fake "edges" of 40-50% that will blow out the bankroll
        # in gray/live mode. If we can't classify the title into a supported
        # market type, skip the market entirely.
        classification = classify_title(title)
        if not classification.is_supported:
            logger.debug(
                "market_type_unsupported",
                market_id=market_id[:20],
                type=classification.type,
                title=title[:60],
            )
            return None

        # Parse dates — preserve TZ-aware versions for historical price lookup,
        # but strip timezone for MarketQuote (which uses naive UTC internally).
        # CRITICAL: prefer `eventStartTime` (actual trading window open) over
        # `startDate` (market listing time, which can be 24h earlier). Using
        # `startDate` fetches BTC's price at listing, not at window-open, which
        # creates a false strike and produces bogus edges.
        end_iso = m.get("endDate") or m.get("endTime")
        start_iso = m.get("eventStartTime") or m.get("startDate") or m.get("startTime")
        try:
            window_end_aware = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            window_end = window_end_aware.replace(tzinfo=None)
        except (ValueError, AttributeError):
            return None
        try:
            if start_iso:
                window_start_aware = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                window_start = window_start_aware.replace(tzinfo=None)
            else:
                window_start_aware = window_end_aware - timedelta(minutes=5)
                window_start = window_end - timedelta(minutes=5)
        except ValueError:
            window_start_aware = window_end_aware - timedelta(minutes=5)
            window_start = window_end - timedelta(minutes=5)

        # Get CLOB token IDs
        clob_tokens = m.get("clobTokenIds")
        if isinstance(clob_tokens, str):
            try:
                clob_tokens = json.loads(clob_tokens)
            except Exception:
                clob_tokens = None
        if not isinstance(clob_tokens, list) or len(clob_tokens) < 2:
            return None

        yes_token, no_token = clob_tokens[0], clob_tokens[1]
        yes_mid, no_mid = await asyncio.gather(
            self._clob_midpoint(yes_token),
            self._clob_midpoint(no_token),
            return_exceptions=True,
        )
        # Handle exceptions
        if isinstance(yes_mid, Exception) or yes_mid is None:
            return None
        if isinstance(no_mid, Exception) or no_mid is None:
            # If NO is missing, infer from YES (they should sum to 1)
            no_mid = 1.0 - yes_mid

        # Clamp to sane range
        yes_price = max(0.01, min(0.99, yes_mid))
        no_price = max(0.01, min(0.99, no_mid))

        # Strike resolution dispatches on market type:
        #   UP_OR_DOWN    → strike = historical BTC spot at window_start (old logic)
        #   FIXED_STRIKE  → strike = parsed dollar amount from title ($72,000 etc)
        ref: float | None = None

        if classification.type == "fixed_strike" and classification.strike is not None:
            # Strike IS the dollar amount in the title. No historical lookup needed.
            ref = classification.strike
            if market_id not in self._ref_price_cache:
                self._ref_price_cache[market_id] = (ref, datetime.utcnow())
                logger.info(
                    "ref_price_from_title",
                    market=market_id[:30],
                    strike=ref,
                    direction=classification.direction,
                    title=title[:60],
                )
        else:
            # Up-or-Down market. Use historical BTC at window_start.
            ref = self.get_ref_price(market_id)
            if ref is None:
                historical = await historical_price(
                    "BTC-USD", at=window_start_aware, http=self._http,
                )
                if historical is not None:
                    self.record_ref_price(market_id, historical)
                    ref = historical
                    logger.info(
                        "ref_price_from_history",
                        market=market_id[:30],
                        window_start=window_start.isoformat()[:19],
                        price=round(historical, 2),
                    )
                elif current_spot is not None:
                    # Fallback: only useful for markets that JUST opened.
                    self.record_ref_price(market_id, current_spot)
                    ref = current_spot
                    logger.warning(
                        "ref_price_fallback_current_spot",
                        market=market_id[:30],
                        spot=round(current_spot, 2),
                    )

        if ref is None:
            return None

        return MarketQuote(
            market_id=market_id,
            title=title,
            yes_price=yes_price,
            no_price=no_price,
            window_start=window_start,
            window_end=window_end,
            starting_ref_price=ref,
            market_type=classification.type,
            strike_direction=classification.direction,
        )

    async def _clob_midpoint(self, token_id: str) -> float | None:
        try:
            resp = await self._http.get(
                CLOB_MIDPOINT_URL,
                params={"token_id": token_id},
                timeout=3.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            mid = data.get("mid")
            return float(mid) if mid is not None else None
        except Exception:
            return None
