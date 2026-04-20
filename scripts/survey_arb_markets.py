"""Survey both platforms for potential arbitrage pair candidates.

For each platform, fetches active markets and groups them by category to make
manual pair-matching easier. Focuses on markets with MECHANICAL resolution:
    - Crypto price-at-close (BTC, ETH with explicit oracle)
    - Fed rate decisions
    - Election state/outcome calls
    - Economic data releases (CPI, PPI, NFP)
    - Sports game outcomes with official league results

Deliberately skips:
    - Ambiguous "by end of year" geopolitical markets
    - Opinion/social polling markets
    - Markets without a clear end_date

Output: structured report to stdout. Pipe to a file for inspection.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on sys.path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

# --- Polymarket (Gamma API) ------------------------------------------------

GAMMA_URL = "https://gamma-api.polymarket.com/markets"

# Category keywords for grouping
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("crypto", ["bitcoin", "btc", "ethereum", "eth ", "solana", "crypto", "stablecoin"]),
    ("fed",    ["fed ", "fomc", "federal reserve", "rate cut", "rate hike", "interest rate"]),
    ("econ",   ["cpi", "ppi", "nonfarm", "jobs report", "unemployment", "gdp", "inflation"]),
    ("oil",    ["oil price", "opec", "crude", "wti", "brent"]),
    ("election", ["election", "senate", "house", "congress", "president", "governor", "midterm"]),
    ("sports", ["nba finals", "nhl", "stanley cup", "super bowl", "world series", "uefa", "champions league", "world cup"]),
    ("weather", ["temperature", "high in", "low in", "snowfall", "hurricane"]),
]


def categorize(title: str) -> str:
    t = title.lower()
    for cat, kws in CATEGORY_RULES:
        if any(kw in t for kw in kws):
            return cat
    return "other"


async def fetch_polymarket_active(client: httpx.AsyncClient) -> list[dict]:
    """Multi-sort fetch to pull different slices of the 500-market active list."""
    seen: set = set()
    out: list[dict] = []
    for order, asc in [("endDate", "true"), ("endDate", "false"), ("volume24hr", "false"), ("liquidityNum", "false")]:
        params = {
            "active": "true",
            "closed": "false",
            "limit": "500",
            "order": order,
            "ascending": asc,
        }
        try:
            resp = await client.get(GAMMA_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  gamma fetch error ({order}/{asc}): {e}")
            continue
        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        for m in markets:
            mid = m.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                out.append(m)
    return out


def is_future_dated(m: dict, max_days_ahead: int = 60) -> bool:
    end_iso = m.get("endDate") or m.get("endTime")
    if not end_iso:
        return False
    try:
        end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    except ValueError:
        return False
    now = datetime.now(timezone.utc)
    return now < end < now + timedelta(days=max_days_ahead)


# --- Kalshi ---------------------------------------------------------------

async def fetch_kalshi_active(settings) -> list[dict]:
    """Use the existing KalshiClient (demo or prod per config)."""
    from src.clients.kalshi_client import KalshiClient
    client = KalshiClient(
        api_key=settings.kalshi.api_key,
        private_key_path=settings.kalshi.private_key_path,
        use_demo=settings.kalshi.use_demo,
        live_mode=False,
    )
    # Go deeper than the default via direct REST calls
    results: list[dict] = []
    try:
        # Raw call to get full market objects (not just UnifiedMarket projection)
        cursor: str | None = None
        fetched = 0
        while fetched < 1000:
            params: dict = {"limit": 200, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            data = await client._request("GET", "/trade-api/v2/markets", params=params)
            markets = data.get("markets", [])
            if not markets:
                break
            results.extend(markets)
            fetched += len(markets)
            cursor = data.get("cursor")
            if not cursor:
                break
    except Exception as e:
        print(f"  kalshi fetch error: {e}")
    finally:
        await client.close()
    return results


# --- Main survey ----------------------------------------------------------

async def main():
    from config.settings import load_settings
    settings = load_settings()

    print("=" * 72)
    print("Surveying Polymarket + Kalshi for potential arbitrage pairs")
    print(f"Now (UTC): {datetime.now(timezone.utc).isoformat()[:19]}")
    print(f"Kalshi config: {'DEMO' if settings.kalshi.use_demo else 'PROD'}")
    print("=" * 72)

    async with httpx.AsyncClient(timeout=20.0) as http:
        print("\n[1/2] Fetching Polymarket...")
        poly_raw = await fetch_polymarket_active(http)
        print(f"  Total unique active: {len(poly_raw)}")
        poly_future = [m for m in poly_raw if is_future_dated(m, max_days_ahead=60)]
        print(f"  Resolving in next 60 days: {len(poly_future)}")

        print("\n[2/2] Fetching Kalshi...")
        kalshi_raw = await fetch_kalshi_active(settings)
        print(f"  Total active: {len(kalshi_raw)}")

    # Categorize Polymarket
    poly_by_cat: dict[str, list[dict]] = defaultdict(list)
    for m in poly_future:
        title = m.get("question") or m.get("title") or ""
        cat = categorize(title)
        poly_by_cat[cat].append(m)

    # Categorize Kalshi — use ticker + title
    kalshi_by_cat: dict[str, list[dict]] = defaultdict(list)
    for m in kalshi_raw:
        title = m.get("title") or m.get("subtitle") or ""
        ticker = m.get("ticker") or ""
        key_text = f"{title} {ticker}"
        cat = categorize(key_text)
        kalshi_by_cat[cat].append(m)

    # Report by category
    print("\n" + "=" * 72)
    print("SUMMARY BY CATEGORY")
    print("=" * 72)
    print(f"{'Category':<12} | {'Polymarket':>10} | {'Kalshi':>8}")
    print("-" * 34)
    for cat in sorted(set(poly_by_cat.keys()) | set(kalshi_by_cat.keys())):
        p = len(poly_by_cat.get(cat, []))
        k = len(kalshi_by_cat.get(cat, []))
        star = " ★" if p > 0 and k > 0 else ""
        print(f"{cat:<12} | {p:>10} | {k:>8}{star}")

    # Show top candidates per high-value category
    print("\n" + "=" * 72)
    print("HIGH-VALUE CATEGORIES (samples)")
    print("=" * 72)

    priority_cats = ["crypto", "fed", "econ", "oil", "election"]
    for cat in priority_cats:
        p_markets = poly_by_cat.get(cat, [])[:10]
        k_markets = kalshi_by_cat.get(cat, [])[:10]
        if not p_markets and not k_markets:
            continue
        print(f"\n--- {cat.upper()} ({len(poly_by_cat.get(cat, []))} poly / {len(kalshi_by_cat.get(cat, []))} kalshi) ---")

        if p_markets:
            print("  POLYMARKET:")
            for m in p_markets:
                title = (m.get("question") or "")[:70]
                end = (m.get("endDate") or "")[:10]
                cid = m.get("conditionId", "")[:12]
                print(f"    [{end}] {title}")
                print(f"      id={cid}  slug={m.get('slug', '')[:40]}")

        if k_markets:
            print("  KALSHI:")
            for m in k_markets:
                title = (m.get("title") or "")[:70]
                close = (m.get("close_time") or m.get("expected_expiration_time") or "")[:10]
                ticker = m.get("ticker", "")
                print(f"    [{close}] {title}")
                print(f"      ticker={ticker}")

    # Dump raw data for offline analysis
    survey_out = Path("data/arb_survey.json")
    survey_out.parent.mkdir(exist_ok=True)
    with survey_out.open("w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "polymarket_count": len(poly_future),
                "kalshi_count": len(kalshi_raw),
                "polymarket_by_cat": {
                    cat: [
                        {
                            "condition_id": m.get("conditionId"),
                            "slug": m.get("slug"),
                            "title": m.get("question"),
                            "end_date": m.get("endDate"),
                            "event_start_time": m.get("eventStartTime"),
                        }
                        for m in ms
                    ]
                    for cat, ms in poly_by_cat.items()
                },
                "kalshi_by_cat": {
                    cat: [
                        {
                            "ticker": m.get("ticker"),
                            "title": m.get("title"),
                            "subtitle": m.get("subtitle"),
                            "close_time": m.get("close_time"),
                            "expected_expiration_time": m.get("expected_expiration_time"),
                            "rules_primary": (m.get("rules_primary") or "")[:200],
                        }
                        for m in ms
                    ]
                    for cat, ms in kalshi_by_cat.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\n\nFull data saved to {survey_out}")
    print("Review categories marked ★ above — those have candidates on both platforms.")


if __name__ == "__main__":
    asyncio.run(main())
