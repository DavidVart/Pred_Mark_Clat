"""Classify every active Polymarket BTC market by type.

We need to distinguish:
    (A) "Up or Down" markets   → strike == BTC price at window open
    (B) "Above $X" markets     → strike == $X parsed from title
    (C) Anything else (we should probably skip)

For each active BTC market ending in the next 24h, print:
    - type guess
    - parsed strike (if fixed-strike)
    - CLOB midpoint
    - time-to-resolution
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_MID = "https://clob.polymarket.com/midpoint"

# --- Type detection regexes ---
UP_OR_DOWN_RX = re.compile(r"bitcoin.*up\s*or\s*down", re.IGNORECASE)
ABOVE_STRIKE_RX = re.compile(
    r"(?:bitcoin|btc).*?(?:above|below|higher than|over|under|reach|hit)\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:k|K)?",
    re.IGNORECASE,
)
# Also handles "Will BTC be above $68,000 on April 17?"


def detect_type_and_strike(title: str) -> tuple[str, float | None]:
    """Return (type, strike_or_none).

    type is one of: "up_or_down" | "fixed_strike" | "unknown"
    strike is the parsed dollar amount for fixed_strike markets, else None.
    """
    if UP_OR_DOWN_RX.search(title):
        return "up_or_down", None
    m = ABOVE_STRIKE_RX.search(title)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            return "unknown", None
        # If the title said "150k" the regex captures "150" then "k" — scan for k
        if re.search(r"\d\s*[kK]\b", title):
            val *= 1000
        return "fixed_strike", val
    return "unknown", None


async def fetch_btc_markets(client: httpx.AsyncClient) -> list[dict]:
    seen = set()
    out = []
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=24)
    for order, asc in [("endDate", "true"), ("endDate", "false"), ("volume24hr", "false")]:
        params = {
            "active": "true",
            "closed": "false",
            "limit": "500",
            "order": order,
            "ascending": asc,
        }
        resp = await client.get(GAMMA_URL, params=params)
        data = resp.json()
        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        for m in markets:
            if m.get("id") in seen:
                continue
            title = (m.get("question") or "").lower()
            if "bitcoin" not in title and "btc" not in title:
                continue
            # Check date in future
            end_iso = m.get("endDate")
            if not end_iso:
                continue
            try:
                end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            except ValueError:
                continue
            if end < now or end > cutoff:
                continue
            seen.add(m.get("id"))
            out.append(m)
    return out


async def clob_midpoint(client: httpx.AsyncClient, token_id: str) -> float | None:
    try:
        r = await client.get(CLOB_MID, params={"token_id": token_id}, timeout=5.0)
        if r.status_code == 200:
            return float(r.json().get("mid", 0))
    except Exception:
        pass
    return None


async def main():
    async with httpx.AsyncClient(timeout=15.0) as client:
        markets = await fetch_btc_markets(client)
        print(f"Found {len(markets)} active BTC markets resolving within 24h\n")

        by_type: dict[str, list[dict]] = {"up_or_down": [], "fixed_strike": [], "unknown": []}
        for m in markets:
            title = m.get("question") or ""
            t, strike = detect_type_and_strike(title)
            clob_tokens = m.get("clobTokenIds")
            if isinstance(clob_tokens, str):
                try:
                    clob_tokens = json.loads(clob_tokens)
                except Exception:
                    clob_tokens = None

            mid_yes = None
            if clob_tokens and len(clob_tokens) >= 1:
                mid_yes = await clob_midpoint(client, clob_tokens[0])

            by_type[t].append({
                "title": title,
                "strike": strike,
                "end": m.get("endDate", "")[:19],
                "event_start": m.get("eventStartTime", "")[:19],
                "mid_yes": mid_yes,
            })

        for t, items in by_type.items():
            print(f"=== {t.upper()}: {len(items)} markets ===")
            for item in items[:15]:
                strike_str = f"strike=${item['strike']:>9,.0f}" if item["strike"] else "strike=?         "
                mid_str = f"mid_yes={item['mid_yes']:.3f}" if item["mid_yes"] is not None else "mid_yes=?    "
                print(f"  [{item['end']}] {strike_str} {mid_str}  {item['title'][:70]}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
