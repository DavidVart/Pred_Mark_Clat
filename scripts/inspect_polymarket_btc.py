"""Diagnostic: find LIVE Polymarket 5M BTC markets ending in the next hour,
and probe the CLOB for live prices.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx


async def fetch_gamma(client: httpx.AsyncClient, params: dict) -> list[dict]:
    resp = await client.get("https://gamma-api.polymarket.com/markets", params=params)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return data.get("data", data.get("markets", []))


async def fetch_clob_price(client: httpx.AsyncClient, token_id: str) -> dict | None:
    """Query CLOB for live price on a token."""
    try:
        resp = await client.get(
            "https://clob.polymarket.com/price",
            params={"token_id": token_id, "side": "buy"},
            timeout=5.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"    CLOB error: {e}")
    return None


async def fetch_clob_midpoint(client: httpx.AsyncClient, token_id: str) -> dict | None:
    try:
        resp = await client.get(
            "https://clob.polymarket.com/midpoint",
            params={"token_id": token_id},
            timeout=5.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"    CLOB midpoint error: {e}")
    return None


async def main() -> None:
    now = datetime.now(timezone.utc)
    future_cutoff = now + timedelta(hours=2)
    print(f"Now (UTC): {now.isoformat()}")
    print(f"Searching for markets ending between now and {future_cutoff.isoformat()}\n")

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Try with a larger limit to get more variety
        all_btc = []
        # Strategy 1: order by endDate DESC — get the far-future and narrow down
        for order, asc in [("endDate", "true"), ("endDate", "false"), ("volume24hr", "false")]:
            params = {
                "active": "true",
                "closed": "false",
                "limit": "500",
                "order": order,
                "ascending": asc,
            }
            markets = await fetch_gamma(client, params)
            for m in markets:
                title = (m.get("question") or m.get("title") or "").lower()
                if "btc" in title or "bitcoin" in title:
                    if m.get("id") not in {x.get("id") for x in all_btc}:
                        all_btc.append(m)

        print(f"Unique BTC markets across queries: {len(all_btc)}\n")

        # Filter to ones ending in the next 2 hours
        live = []
        for m in all_btc:
            end_iso = m.get("endDate") or m.get("endTime")
            if not end_iso:
                continue
            try:
                end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            except Exception:
                continue
            if now <= end <= future_cutoff:
                live.append((end, m))

        live.sort(key=lambda t: t[0])
        print(f"LIVE BTC markets ending in the next 2h: {len(live)}\n")

        for end, m in live[:10]:
            title = m.get("question") or m.get("title") or ""
            slug = m.get("slug") or ""
            op = m.get("outcomePrices")
            if isinstance(op, str):
                try:
                    op = json.loads(op)
                except Exception:
                    pass
            clob_tokens = m.get("clobTokenIds")
            if isinstance(clob_tokens, str):
                try:
                    clob_tokens = json.loads(clob_tokens)
                except Exception:
                    pass

            print(f"  {end.isoformat()[:19]}  {title[:70]}")
            print(f"     slug: {slug[:50]}")
            print(f"     outcomePrices: {op}")
            print(f"     clobTokenIds: {clob_tokens}")

            if clob_tokens and len(clob_tokens) >= 2:
                yes_price = await fetch_clob_price(client, clob_tokens[0])
                yes_mid = await fetch_clob_midpoint(client, clob_tokens[0])
                print(f"     CLOB YES price (buy): {yes_price}")
                print(f"     CLOB YES midpoint:    {yes_mid}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
