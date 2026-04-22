"""Weather edge recorder — continuously logs Kalshi prices + our NOAA predictions.

Every POLL_INTERVAL (default 15 min):
  - Fetch active Kalshi weather markets for 5 cities
  - Fetch NOAA hourly forecasts (cached per city for 30 min)
  - For each market, compute our predicted YES probability
  - Log snapshot to DB

On a slower schedule, checks for resolved markets and backfills outcomes.
"""

from __future__ import annotations

import asyncio
import math
import re
import signal as os_signal
from datetime import datetime, timedelta, timezone

import httpx

from src.utils.logging import get_logger, setup_logging
from src.weather_recorder.db import WeatherRecorderDB

logger = get_logger("weather_recorder")


CITIES = {
    "KXHIGHNY":  ("NYC", 40.7794, -73.9691),
    "KXHIGHLAX": ("LAX", 33.9416, -118.4085),
    "KXHIGHCHI": ("CHI", 41.9786, -87.9048),
    "KXHIGHMIA": ("MIA", 25.7617, -80.1918),
    "KXHIGHAUS": ("AUS", 30.2672, -97.7431),
}

DEFAULT_SIGMA_F = 3.0  # assumed 1-sigma NOAA 24h forecast error
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


# ---------- Math ----------

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_leq(threshold: float, forecast: float, sigma: float) -> float:
    if forecast is None:
        return 0.5
    return _normal_cdf((threshold - forecast) / max(0.1, sigma))


def parse_strike(ticker: str, subtitle: str) -> tuple[str, float] | None:
    m_t = re.search(r"-T(\d+(?:\.\d+)?)", ticker)
    if m_t:
        threshold = float(m_t.group(1))
        direction = "above"
        if "below" in subtitle.lower() or "or below" in subtitle.lower():
            direction = "below"
        return (direction, threshold)
    m_b = re.search(r"-B(\d+(?:\.\d+)?)", ticker)
    if m_b:
        return ("between", float(m_b.group(1)))
    return None


def parse_event_date(event_ticker: str) -> str | None:
    m = re.search(r"(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if not m:
        return None
    yy, mon, dd = m.groups()
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    try:
        return f"20{yy}-{months[mon]:02d}-{int(dd):02d}"
    except (KeyError, ValueError):
        return None


def compute_our_prob(direction: str, threshold: float, noaa_high: float, sigma: float) -> float:
    if direction == "below":
        return prob_leq(threshold, noaa_high, sigma)
    elif direction == "above":
        return 1.0 - prob_leq(threshold, noaa_high, sigma)
    elif direction == "between":
        low = threshold - 0.5
        high = threshold + 0.5
        return prob_leq(high, noaa_high, sigma) - prob_leq(low, noaa_high, sigma)
    return 0.5


# ---------- HTTP ----------

async def kalshi_markets(client: httpx.AsyncClient, series: str) -> list[dict]:
    r = await client.get(
        f"{KALSHI_BASE}/markets",
        params={"series_ticker": series, "limit": 100, "status": "open"},
    )
    if r.status_code != 200:
        logger.warning("kalshi_markets_error", series=series, status=r.status_code)
        return []
    return r.json().get("markets", [])


async def kalshi_orderbook(client: httpx.AsyncClient, ticker: str) -> dict | None:
    try:
        r = await client.get(f"{KALSHI_BASE}/markets/{ticker}/orderbook", timeout=8.0)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("orderbook", data.get("orderbook_fp"))
    except Exception:
        return None


async def noaa_hourly(client: httpx.AsyncClient, lat: float, lon: float) -> dict | None:
    try:
        r = await client.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            headers={"User-Agent": "pmc-weather-recorder/1.0"},
            timeout=10.0,
        )
        if r.status_code != 200:
            return None
        forecast_url = r.json()["properties"]["forecastHourly"]
        r2 = await client.get(
            forecast_url,
            headers={"User-Agent": "pmc-weather-recorder/1.0"},
            timeout=10.0,
        )
        if r2.status_code != 200:
            return None
        return r2.json()
    except Exception as e:
        logger.warning("noaa_fetch_error", error=str(e))
        return None


def daily_high_from_forecast(forecast_json: dict, target_date: str) -> float | None:
    if not forecast_json:
        return None
    periods = forecast_json.get("properties", {}).get("periods", [])
    temps = [p.get("temperature") for p in periods if (p.get("startTime") or "")[:10] == target_date]
    temps = [t for t in temps if t is not None]
    return max(temps) if temps else None


# ---------- Book parsing ----------

def parse_book(book: dict) -> tuple[float | None, float | None, float]:
    """Return (yes_best_bid, yes_best_ask, book_depth_size)."""
    yes_orders = book.get("yes", []) or book.get("yes_dollars", [])
    no_orders = book.get("no", []) or book.get("no_dollars", [])
    yes_best_bid = None
    yes_best_ask = None
    depth = 0.0
    try:
        if yes_orders:
            yes_best_bid = max(float(o[0]) for o in yes_orders)
        if no_orders:
            no_best_bid = max(float(o[0]) for o in no_orders)
            yes_best_ask = 1.0 - no_best_bid
        for o in yes_orders:
            depth += float(o[1])
        for o in no_orders:
            depth += float(o[1])
    except (ValueError, TypeError):
        pass
    return yes_best_bid, yes_best_ask, depth


# ---------- Main recorder loop ----------

async def record_once(client: httpx.AsyncClient, db: WeatherRecorderDB, noaa_cache: dict) -> int:
    """Single pass: snapshot all active weather markets + our predictions."""
    snaps_logged = 0
    for series_ticker, (city, lat, lon) in CITIES.items():
        forecast = noaa_cache.get(series_ticker)
        if forecast is None:
            forecast = await noaa_hourly(client, lat, lon)
            noaa_cache[series_ticker] = forecast
        if forecast is None:
            logger.warning("no_forecast_skipping_city", city=city)
            continue

        markets = await kalshi_markets(client, series_ticker)
        for m in markets:
            ticker = m["ticker"]
            subtitle = m.get("subtitle", "")
            parsed = parse_strike(ticker, subtitle)
            if not parsed:
                continue
            direction, threshold = parsed

            event_ticker = m.get("event_ticker") or "-".join(ticker.split("-")[:2])
            event_date = parse_event_date(event_ticker)
            if not event_date:
                continue

            noaa_high = daily_high_from_forecast(forecast, event_date)
            if noaa_high is None:
                continue

            our_prob = compute_our_prob(direction, threshold, noaa_high, DEFAULT_SIGMA_F)

            book = await kalshi_orderbook(client, ticker)
            yes_bid = yes_ask = None
            depth = 0.0
            if book:
                yes_bid, yes_ask, depth = parse_book(book)
            yes_mid = (yes_bid + yes_ask) / 2 if yes_bid is not None and yes_ask is not None else None

            await db.log_snapshot({
                "market_ticker": ticker,
                "event_ticker": event_ticker,
                "event_date": event_date,
                "city": city,
                "direction": direction,
                "threshold": threshold,
                "yes_best_bid": yes_bid,
                "yes_best_ask": yes_ask,
                "yes_mid": yes_mid,
                "yes_book_depth": depth,
                "our_yes_prob": our_prob,
                "noaa_high_forecast": noaa_high,
                "noaa_sigma": DEFAULT_SIGMA_F,
            })
            snaps_logged += 1

    return snaps_logged


async def check_resolutions(client: httpx.AsyncClient, db: WeatherRecorderDB) -> int:
    """Find unresolved snapshots whose event_date is yesterday or earlier,
    and backfill the outcome from Kalshi's market result."""
    cursor = await db.db.execute(
        """SELECT DISTINCT market_ticker, direction, threshold, event_date
           FROM wr_snapshots
           WHERE resolved = 0
             AND event_date <= date('now', '-0 days')
           ORDER BY event_date"""
    )
    pending = await cursor.fetchall()
    resolved_count = 0
    for row in pending:
        ticker = row["market_ticker"]
        try:
            r = await client.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=8.0)
            if r.status_code != 200:
                continue
            data = r.json()
            m = data.get("market", data)
            status = m.get("status")
            result = m.get("result")
            if status not in ("finalized", "settled") or not result:
                continue
            outcome_yes = 1 if result.lower() in ("yes", "1", "true") else 0
            # actual_high is harder to get — we might need to derive from threshold + outcome
            # For binary-below: actual <= threshold iff YES → use threshold as upper bound
            # Skipped: just record outcome
            actual_high = 0.0
            n = await db.mark_resolved(ticker, actual_high, outcome_yes)
            resolved_count += n
            logger.info("market_resolved", ticker=ticker, outcome_yes=outcome_yes, snapshots=n)
        except Exception as e:
            logger.debug("resolution_check_error", ticker=ticker, error=str(e))
    return resolved_count


async def run_recorder(
    db_path: str = "weather_recorder.db",
    poll_interval_sec: int = 15 * 60,       # 15 min
    resolution_check_sec: int = 60 * 60,    # 1 hour
    log_level: str = "INFO",
) -> None:
    setup_logging(log_level)
    logger.info("weather_recorder_starting", poll_sec=poll_interval_sec, db=db_path)

    db = WeatherRecorderDB(db_path)
    await db.initialize()

    shutdown_event = asyncio.Event()

    def _handle_signal(sig, frame):
        logger.info("shutdown_signal", sig=sig)
        shutdown_event.set()

    os_signal.signal(os_signal.SIGINT, _handle_signal)
    os_signal.signal(os_signal.SIGTERM, _handle_signal)

    async with httpx.AsyncClient(timeout=15.0) as client:
        last_resolution_check = 0.0
        try:
            while not shutdown_event.is_set():
                cycle_start = datetime.now(timezone.utc)

                # NOAA forecasts change slowly; refresh every 30 min.
                # Simple approach: rebuild cache each cycle (5 cities × ~0.5s each = 2.5s).
                noaa_cache: dict = {}

                try:
                    n = await record_once(client, db, noaa_cache)
                    logger.info("snapshot_recorded", snapshots=n,
                                cycle_ms=int((datetime.now(timezone.utc) - cycle_start).total_seconds() * 1000))
                except Exception as e:
                    logger.error("snapshot_error", error=str(e))

                # Check for resolutions hourly
                import time as _time
                now_mono = _time.monotonic()
                if now_mono - last_resolution_check > resolution_check_sec:
                    last_resolution_check = now_mono
                    try:
                        r = await check_resolutions(client, db)
                        if r > 0:
                            logger.info("resolutions_backfilled", count=r)
                    except Exception as e:
                        logger.error("resolution_check_error", error=str(e))

                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval_sec)
                except asyncio.TimeoutError:
                    pass
        finally:
            await db.close()
            logger.info("weather_recorder_stopped")
