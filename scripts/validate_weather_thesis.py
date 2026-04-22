"""Validate the Kalshi weather trading thesis in ~30 minutes.

The thesis: Kalshi weather markets derive from public NOAA forecast data.
A bot that ingests forecast updates should detect meaningful gaps between
current Kalshi pricing and NOAA consensus.

This script measures:
  1. How many active Kalshi weather markets exist right now?
  2. For each, what's the implied most-likely temperature?
  3. What does NOAA (HRRR or NDFD) forecast for the same location/time?
  4. Is there a measurable gap we could trade?

If we see consistent, meaningful gaps → thesis holds, build the bot.
If Kalshi is tight-aligned with NOAA (within forecast uncertainty) → thesis
dead, pivot to something else.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


async def get_kalshi_weather_markets(client: httpx.AsyncClient) -> list[dict]:
    """Kalshi weather series tickers (from our earlier series scan)."""
    # KXHIGHNY, KXHIGHLAX, KXHIGHCHI — NYC/LA/Chicago daily highs
    all_markets = []
    for series in ["KXHIGHNY", "KXHIGHLAX", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS"]:
        try:
            r = await client.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params={"series_ticker": series, "limit": 100, "status": "open"},
            )
            if r.status_code != 200:
                print(f"  {series}: status {r.status_code}")
                continue
            data = r.json()
            markets = data.get("markets", [])
            for m in markets:
                m["_series"] = series
            all_markets.extend(markets)
            print(f"  {series}: {len(markets)} open markets")
        except Exception as e:
            print(f"  {series}: error {e}")
    return all_markets


async def noaa_ndfd_forecast(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
) -> dict | None:
    """NOAA National Digital Forecast Database via weather.gov API.

    Free, no API key. Returns daily high temperatures for the next ~7 days.
    """
    try:
        # 1. Get grid identifier for the location
        r = await client.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            headers={"User-Agent": "pmc-newsalpha-investigation/1.0"},
            timeout=10.0,
        )
        if r.status_code != 200:
            return None
        office = r.json()["properties"]
        forecast_url = office["forecastHourly"]

        # 2. Get hourly forecast
        r2 = await client.get(
            forecast_url,
            headers={"User-Agent": "pmc-newsalpha-investigation/1.0"},
            timeout=10.0,
        )
        if r2.status_code != 200:
            return None
        return r2.json()
    except Exception as e:
        print(f"    NOAA error: {e}")
        return None


def infer_city_from_ticker(ticker: str) -> tuple[str, float, float] | None:
    """Map Kalshi series ticker to lat/lon."""
    mapping = {
        "KXHIGHNY": ("New York (Central Park)", 40.7794, -73.9691),
        "KXHIGHLAX": ("Los Angeles (LAX)", 33.9416, -118.4085),
        "KXHIGHCHI": ("Chicago (O'Hare)", 41.9786, -87.9048),
        "KXHIGHMIA": ("Miami", 25.7617, -80.1918),
        "KXHIGHAUS": ("Austin", 30.2672, -97.7431),
    }
    for k, v in mapping.items():
        if ticker.startswith(k):
            return v
    return None


def extract_kalshi_temp_threshold(ticker: str, subtitle: str) -> float | None:
    """Extract the temperature threshold from Kalshi's subtitle or ticker.

    E.g., ticker='KXHIGHNY-26APR22-T65' subtitle='65° or above' → 65
    """
    # Try the ticker tail first (e.g., 'T65')
    import re
    m = re.search(r"-T(\d+)(?:\.\d+)?", ticker)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+)°", subtitle)
    if m2:
        return float(m2.group(1))
    return None


async def main():
    print("=" * 70)
    print("KALSHI WEATHER THESIS VALIDATION")
    print("=" * 70)
    print()

    async with httpx.AsyncClient(timeout=20.0) as client:
        print("[1/3] Fetching active Kalshi weather markets...")
        markets = await get_kalshi_weather_markets(client)
        print(f"\nTotal: {len(markets)} open weather markets\n")

        if not markets:
            print("NO WEATHER MARKETS FOUND — thesis is dead as a specific target.")
            print("Possible reasons: Kalshi renamed series, markets closed today, API issue.")
            return

        # Group by event (same date + city = same underlying question)
        by_event: dict[str, list[dict]] = {}
        for m in markets:
            event = m.get("event_ticker", "") or m.get("ticker", "").split("-")[0]
            by_event.setdefault(event, []).append(m)

        print(f"Active events: {len(by_event)}")
        print()

        # Show first few events with their strike ladder
        for event, ms in list(by_event.items())[:5]:
            ms_sorted = sorted(ms, key=lambda x: extract_kalshi_temp_threshold(x.get("ticker", ""), x.get("subtitle", "")) or 0)
            print(f"--- {event} ({len(ms_sorted)} strikes) ---")
            for m in ms_sorted[:20]:
                threshold = extract_kalshi_temp_threshold(m.get("ticker", ""), m.get("subtitle", ""))
                yes_bid = m.get("yes_bid")
                yes_ask = m.get("yes_ask")
                close_time = m.get("close_time", "")[:16]
                subtitle = m.get("subtitle", "")[:40]
                print(f"  {m['ticker']:<35} | close={close_time} | yes={yes_bid}/{yes_ask} | {subtitle}")
            print()

        print("\n[2/3] Fetching NOAA forecast for NYC...")
        noaa_data = await noaa_ndfd_forecast(client, 40.7794, -73.9691)
        if noaa_data is None:
            print("    NOAA API failed — try again or use different source.")
        else:
            periods = noaa_data.get("properties", {}).get("periods", [])
            print(f"    Got {len(periods)} hourly forecast periods")
            if periods:
                print(f"    First period: {periods[0].get('startTime', '')[:16]} → {periods[0].get('temperature')}°{periods[0].get('temperatureUnit')}")
                print(f"    Last period:  {periods[-1].get('startTime', '')[:16]} → {periods[-1].get('temperature')}°{periods[-1].get('temperatureUnit')}")

                # Find max temp over next 18 hours (today's daily high window)
                import datetime
                now = datetime.datetime.utcnow()
                today_hours = [
                    p for p in periods
                    if datetime.datetime.fromisoformat(p["startTime"].replace("Z", "+00:00")).replace(tzinfo=None).date() == now.date()
                ]
                if today_hours:
                    max_temp = max(p["temperature"] for p in today_hours)
                    print(f"    NOAA daily high forecast for NYC today: {max_temp}°F")

        print("\n[3/3] Cross-referencing NYC markets vs NOAA forecast...")
        # Find today's KXHIGHNY markets
        today_ny = [
            m for m in markets
            if m.get("ticker", "").startswith("KXHIGHNY")
        ][:20]
        if today_ny:
            print(f"    {len(today_ny)} NYC high temperature markets to analyze")
            print("    For each strike, what's the implied probability?")
            for m in sorted(today_ny, key=lambda x: extract_kalshi_temp_threshold(x.get("ticker", ""), x.get("subtitle", "")) or 0):
                threshold = extract_kalshi_temp_threshold(m.get("ticker", ""), m.get("subtitle", ""))
                if threshold is None:
                    continue
                yes_bid = m.get("yes_bid")
                if yes_bid is None:
                    continue
                # Kalshi returns price in cents
                try:
                    implied_prob = yes_bid / 100.0
                except Exception:
                    implied_prob = None
                print(f"    threshold={threshold}°F yes_bid={yes_bid} implied_prob={implied_prob}")

        print()
        print("=" * 70)
        print("THESIS CHECK:")
        print("  - Are there multiple strikes per event? (need for ladder math)")
        print("  - Are prices posted or is book empty?")
        print("  - Is NOAA forecast reachable without paid access?")
        print("  - Does Kalshi's implied ladder look aligned with NOAA consensus?")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
