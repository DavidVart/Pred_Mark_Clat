"""Connectivity smoke test — verify all external APIs work with current credentials."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from config.settings import load_settings
from src.clients.kalshi_client import KalshiClient
from src.clients.openrouter_client import OpenRouterClient
from src.clients.polymarket_client import PolymarketClient


async def test_polymarket():
    print("\n=== Polymarket (Gamma API) ===")
    settings = load_settings()
    client = PolymarketClient(
        wallet_private_key=settings.polymarket.wallet_private_key,
        live_mode=False,
    )
    try:
        markets = await client.get_active_markets(limit=5)
        print(f"OK — fetched {len(markets)} active markets")
        for m in markets[:3]:
            print(f"  - {m.title[:70]} | YES: {m.yes_price:.2f} | vol: {m.volume:,}")
    except Exception as e:
        print(f"FAILED: {e}")
    finally:
        await client.close()


async def test_kalshi():
    print("\n=== Kalshi (REST API) ===")
    settings = load_settings()
    client = KalshiClient(
        api_key=settings.kalshi.api_key,
        private_key_path=settings.kalshi.private_key_path,
        use_demo=settings.kalshi.use_demo,
        live_mode=False,
    )
    try:
        markets = await client.get_active_markets(limit=5)
        print(f"OK — fetched {len(markets)} active markets")
        for m in markets[:3]:
            print(f"  - {m.title[:70]} | YES: {m.yes_price:.2f} | vol: {m.volume:,}")

        try:
            balance = await client.get_balance()
            print(f"  Balance: ${balance:.2f}")
        except Exception as e:
            print(f"  Balance check failed: {e}")
    except Exception as e:
        print(f"FAILED: {e}")
    finally:
        await client.close()


async def test_openrouter():
    print("\n=== OpenRouter (LLM gateway) ===")
    settings = load_settings()
    client = OpenRouterClient(settings.openrouter_api_key)
    try:
        text, cost, tokens = await client.complete(
            model="anthropic/claude-sonnet-4",
            messages=[
                {"role": "user", "content": "Respond with exactly 'OK' if you can hear me."}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        print(f"OK — model responded: '{text.strip()}'")
        print(f"  Tokens: {tokens}, Cost: ${cost:.6f}")
    except Exception as e:
        print(f"FAILED: {e}")
    finally:
        await client.close()


async def main():
    print("Connectivity smoke test — this will make ~1 API call per service")
    await test_polymarket()
    await test_kalshi()
    await test_openrouter()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
