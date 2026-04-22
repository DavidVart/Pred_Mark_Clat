"""Measure the actual gap between Kalshi weather market pricing and NOAA forecast.

For each active weather market strike today:
    1. Get Kalshi mid-price from orderbook
    2. Get NOAA forecast for the location
    3. Compute naive probability using forecast ± uncertainty
    4. Compare: is there a systematic gap we could trade?

If gap > typical market spread (~3-5¢) consistently across markets → thesis
validated, edge plausibly exists.
If gap < market spread → market already prices NOAA correctly, no edge.
"""

from __future__ import annotations

import asyncio
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


# --- City mapping with NOAA grid points ---
CITIES = {
    "KXHIGHNY":  ("NYC",     40.7794, -73.9691),
    "KXHIGHLAX": ("LAX",     33.9416, -118.4085),
    "KXHIGHCHI": ("CHI",     41.9786, -87.9048),
    "KXHIGHMIA": ("MIA",     25.7617, -80.1918),
    "KXHIGHAUS": ("AUS",     30.2672, -97.7431),
}


async def kalshi_orderbook(client: httpx.AsyncClient, ticker: str) -> dict | None:
    r = await client.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook")
    if r.status_code != 200:
        return None
    return r.json().get("orderbook", r.json().get("orderbook_fp"))


def best_bid_ask_from_book(book: dict, side: str) -> tuple[float | None, float | None]:
    """Extract best bid and best ask for YES or NO side from Kalshi orderbook.

    Kalshi format: yes_dollars (bids on YES at these prices) + no_dollars (bids on NO).
    Since yes + no sum to 1, ask-side for YES = 1 - highest NO bid.
    """
    yes_bids = book.get("yes", book.get("yes_dollars", [])) if side == "yes" else None
    no_bids = book.get("no", book.get("no_dollars", [])) if side == "yes" else None

    # Alternative format that Kalshi sometimes uses
    if side == "yes":
        yes_orders = book.get("yes", []) or book.get("yes_dollars", [])
        no_orders = book.get("no", []) or book.get("no_dollars", [])
    else:
        yes_orders = book.get("no", []) or book.get("no_dollars", [])
        no_orders = book.get("yes", []) or book.get("yes_dollars", [])

    # Highest bid on our side = best bid
    best_bid = None
    if yes_orders:
        try:
            best_bid = max(float(o[0]) for o in yes_orders)
        except (ValueError, TypeError):
            pass

    # Ask for our side = 1 - highest bid on the OTHER side
    best_ask = None
    if no_orders:
        try:
            other_best_bid = max(float(o[0]) for o in no_orders)
            best_ask = 1.0 - other_best_bid
        except (ValueError, TypeError):
            pass

    return best_bid, best_ask


async def get_weather_markets_by_event(client: httpx.AsyncClient, series: str) -> dict[str, list[dict]]:
    r = await client.get(
        "https://api.elections.kalshi.com/trade-api/v2/markets",
        params={"series_ticker": series, "limit": 100, "status": "open"},
    )
    if r.status_code != 200:
        return {}
    by_event: dict[str, list[dict]] = {}
    for m in r.json().get("markets", []):
        event = m.get("event_ticker", "") or "-".join(m.get("ticker", "").split("-")[:2])
        by_event.setdefault(event, []).append(m)
    return by_event


async def noaa_forecast(client: httpx.AsyncClient, lat: float, lon: float) -> dict | None:
    """Return hourly forecast list (next ~156 hours)."""
    try:
        r = await client.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            headers={"User-Agent": "pmc-newsalpha-investigation/1.0"},
            timeout=10.0,
        )
        if r.status_code != 200:
            return None
        forecast_url = r.json()["properties"]["forecastHourly"]
        r2 = await client.get(
            forecast_url,
            headers={"User-Agent": "pmc-newsalpha-investigation/1.0"},
            timeout=10.0,
        )
        if r2.status_code != 200:
            return None
        return r2.json()
    except Exception:
        return None


def daily_high_forecast(forecast_json: dict, target_date: datetime) -> float | None:
    """Find the max temperature forecast for the target date (LOCAL date).

    NOAA forecasts come in the location's LOCAL timezone already (e.g. EDT for NYC).
    We must match LOCAL date, not UTC, because "daily high" is local-noon concept.
    """
    if not forecast_json:
        return None
    periods = forecast_json.get("properties", {}).get("periods", [])
    temps = []
    target_ymd = target_date.strftime("%Y-%m-%d")
    for p in periods:
        start = p.get("startTime", "")
        # startTime is local-time with offset, e.g. "2026-04-23T08:00:00-04:00"
        # We just match on the leading YYYY-MM-DD (local date of period start)
        if start[:10] == target_ymd:
            temps.append(p.get("temperature", 0))
    return max(temps) if temps else None


def parse_event_date(event_ticker: str) -> datetime | None:
    """Parse date from a Kalshi event ticker like KXHIGHNY-26APR23 → 2026-04-23."""
    import re
    m = re.search(r"(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if not m:
        return None
    yy, mon, dd = m.groups()
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    try:
        return datetime(2000 + int(yy), months[mon], int(dd))
    except (KeyError, ValueError):
        return None


def prob_temp_leq(threshold: float, forecast: float, sigma: float = 3.0) -> float:
    """P(daily high <= threshold) assuming Normal(forecast, sigma) error."""
    if forecast is None:
        return 0.5
    z = (threshold - forecast) / sigma
    # standard normal CDF via erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def parse_strike(ticker: str, subtitle: str) -> tuple[str, float] | None:
    """Return ('above', threshold), ('below', threshold), or ('between', low) from ticker.

    Kalshi weather ticker format:
      -TXX: threshold (above or below X depending on subtitle)
      -BXX.5: between X and X+1
    """
    import re
    # Above/below: Tnn
    m_t = re.search(r"-T(\d+(?:\.\d+)?)", ticker)
    if m_t:
        threshold = float(m_t.group(1))
        if "below" in subtitle.lower() or "or below" in subtitle.lower():
            return ("below", threshold)
        if "above" in subtitle.lower() or "or above" in subtitle.lower():
            return ("above", threshold)
        # Ambiguous — use threshold direction heuristic (Kalshi convention: T<low> = below-threshold, T<high> = above-threshold)
    # Between: Bxx.5
    m_b = re.search(r"-B(\d+(?:\.\d+)?)", ticker)
    if m_b:
        return ("between", float(m_b.group(1)))
    return None


async def analyze():
    now = datetime.now(timezone.utc)

    print("=" * 80)
    print(f"WEATHER EDGE MEASUREMENT | UTC now: {now.isoformat()[:19]}")
    print("=" * 80)
    print()

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Fetch NOAA forecasts ONCE per city (next 7 days of hourlies)
        print("Fetching NOAA forecasts for each city...")
        forecasts = {}
        for series_ticker, (city, lat, lon) in CITIES.items():
            f = await noaa_forecast(client, lat, lon)
            forecasts[series_ticker] = f
            if f:
                periods = f.get("properties", {}).get("periods", [])
                print(f"  {city:<4}: {len(periods)} hourly periods fetched")

        print()
        print(f"{'TICKER':<40} {'DIR':<8} {'TH':<5} {'MID':<6} {'NOAA':<5} {'NOAA-P':<8} {'GAP':<8}")
        print("-" * 95)

        all_gaps: list[float] = []
        for series_ticker, (city, _, _) in CITIES.items():
            noaa_json = forecasts.get(series_ticker)
            if noaa_json is None:
                continue
            by_event = await get_weather_markets_by_event(client, series_ticker)

            for event_ticker, event_markets in by_event.items():
                # Use the event's date, not today's
                event_date = parse_event_date(event_ticker)
                if event_date is None:
                    continue
                noaa_high = daily_high_forecast(noaa_json, event_date)
                if noaa_high is None:
                    # Event might be today and resolution is local "end of day";
                    # use max of remaining hours today
                    continue

                for m in sorted(event_markets, key=lambda x: x["ticker"]):
                    ticker = m["ticker"]
                    subtitle = m.get("subtitle", "")
                    parsed = parse_strike(ticker, subtitle)
                    if not parsed:
                        continue
                    direction, threshold = parsed

                    book = await kalshi_orderbook(client, ticker)
                    if not book:
                        continue
                    bid, ask = best_bid_ask_from_book(book, "yes")
                    mid = (bid + ask) / 2 if bid is not None and ask is not None else None

                    if direction == "below":
                        noaa_yes_prob = prob_temp_leq(threshold, noaa_high)
                    elif direction == "above":
                        noaa_yes_prob = 1.0 - prob_temp_leq(threshold, noaa_high)
                    elif direction == "between":
                        # Between X and X+1: use ±0.5 window, wider sigma to reflect 1°F bin granularity
                        low = threshold - 0.5
                        high = threshold + 0.5
                        noaa_yes_prob = prob_temp_leq(high, noaa_high) - prob_temp_leq(low, noaa_high)
                    else:
                        noaa_yes_prob = 0.5

                    gap = (noaa_yes_prob - mid) if mid is not None else None

                    mid_s = f"{mid:.3f}" if mid is not None else "-"
                    gap_s = f"{gap:+.3f}" if gap is not None else "-"

                    print(f"{ticker:<40} {direction:<8} {threshold:<5} {mid_s:<6} {noaa_high:<5} {noaa_yes_prob:<8.3f} {gap_s}")

                    if gap is not None and mid is not None and 0.05 < mid < 0.95:
                        all_gaps.append(gap)

        print()
        print("=" * 80)
        print(f"TRADEABLE GAPS (markets priced in [0.05, 0.95]): {len(all_gaps)}")
        if all_gaps:
            abs_gaps = [abs(g) for g in all_gaps]
            print(f"  |gap| mean:   {sum(abs_gaps)/len(abs_gaps):.3f}")
            print(f"  |gap| max:    {max(abs_gaps):.3f}")
            big_gaps = [g for g in all_gaps if abs(g) > 0.05]
            print(f"  |gap| > 5%:   {len(big_gaps)}/{len(all_gaps)}  ({100*len(big_gaps)/len(all_gaps):.0f}%)")
        print("=" * 80)
        print()
        print("INTERPRETATION:")
        print("  |gap| < 0.02: market tracks NOAA tightly → no edge, pivot.")
        print("  |gap| 0.02-0.05: possibly tradeable IF fees < gap. Marginal.")
        print("  |gap| > 0.05 on multiple markets: STRONG thesis, build the bot.")


if __name__ == "__main__":
    asyncio.run(analyze())
