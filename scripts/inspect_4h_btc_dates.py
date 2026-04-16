"""What date fields does Polymarket actually expose for the 4h BTC market?"""

from __future__ import annotations

import asyncio
import json

import httpx


async def main() -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": "500",
                "order": "endDate",
                "ascending": "true",
            },
        )
        markets = resp.json()
        if isinstance(markets, dict):
            markets = markets.get("data", markets.get("markets", []))

    found = None
    for m in markets:
        title = (m.get("question") or m.get("title") or "").lower()
        slug = (m.get("slug") or "").lower()
        if "4h" in slug and "btc" in slug and "april 16" in title.lower():
            found = m
            break

    if not found:
        # Fall back to any April 16 4h BTC market
        for m in markets:
            slug = (m.get("slug") or "").lower()
            if "btc-updown-4h" in slug:
                t = m.get("question") or ""
                if "April 16" in t:
                    found = m
                    break

    if not found:
        # Relax: any active BTC "Up or Down" market
        for m in markets:
            title = (m.get("question") or "").lower()
            if "bitcoin" in title and "up or down" in title:
                found = m
                break

    if not found:
        print("No active BTC Up or Down market found")
        return

    print(f"Found: {found.get('question')}")
    print(f"slug: {found.get('slug')}")
    print()
    print("Date-like fields on this market:")
    for k, v in sorted(found.items()):
        if any(s in k.lower() for s in ("date", "time", "start", "end")):
            print(f"  {k}: {v}")
    print()
    print("Full JSON keys: ", sorted(found.keys()))


if __name__ == "__main__":
    asyncio.run(main())
