"""Coinbase Advanced Trade REST API — historical candles / prices.

We use this to recover the TRUE opening BTC price for a market that opened
before we started observing. Without it, snapshotting 'current spot' at first
sighting produces a wrong strike and generates massive false-positive edges
on in-progress markets.

Endpoint:
    GET https://api.exchange.coinbase.com/products/<product>/candles
    params:
      start = ISO-8601 start time (UTC)
      end   = ISO-8601 end time (UTC)
      granularity = seconds per candle (60, 300, 900, 3600, 21600, 86400)

Response: array of candles [time, low, high, open, close, volume],
sorted descending by time.

No auth needed for public market data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from src.utils.logging import get_logger

logger = get_logger("newsalpha.coinbase_rest")

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"


async def historical_price(
    product: str,
    at: datetime,
    http: httpx.AsyncClient | None = None,
    granularity: int = 60,
) -> float | None:
    """Return the open price of the 1-minute candle containing `at`.

    If `at` is TZ-naive it's treated as UTC. Returns None on any failure so
    the caller can fall back gracefully.
    """
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)

    # Pull a tight window around `at` (one candle either side so we definitely
    # get coverage even if Coinbase aligns candles off-minute).
    start = at - timedelta(seconds=granularity)
    end = at + timedelta(seconds=granularity * 2)

    params = {
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "granularity": granularity,
    }
    url = COINBASE_CANDLES_URL.format(product=product)

    owned_client = http is None
    client = http or httpx.AsyncClient(timeout=8.0)
    try:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            logger.warning("candles_http_error", status=resp.status_code, body=resp.text[:200])
            return None
        data = resp.json()
    except Exception as e:
        logger.warning("candles_fetch_error", error=str(e))
        return None
    finally:
        if owned_client:
            await client.aclose()

    if not isinstance(data, list) or not data:
        return None

    # Find the candle whose time bucket contains `at`.
    at_epoch = at.timestamp()
    best = None
    best_diff = float("inf")
    for row in data:
        # Coinbase returns [time, low, high, open, close, volume]
        if not row or len(row) < 5:
            continue
        candle_time = row[0]  # epoch seconds of candle OPEN
        diff = abs(candle_time - at_epoch)
        if diff < best_diff:
            best_diff = diff
            best = row

    if best is None:
        return None

    try:
        open_price = float(best[3])
        return open_price
    except (TypeError, ValueError):
        return None
