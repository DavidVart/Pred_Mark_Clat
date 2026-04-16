"""Coinbase Advanced Trade public websocket client for BTC/ETH spot.

Uses the free, unauthenticated public feed at wss://ws-feed.exchange.coinbase.com.
Subscribes to the 'ticker' channel which pushes a price update on every trade.

Design:
    - Maintains the latest price in-memory (get_price(symbol) is O(1))
    - Auto-reconnects with exponential backoff
    - Optional callback on each tick for custom processing
    - Does NOT write to DB by default — caller opts in

Why not use a library?
    Popular ones (coinbase-advanced-py, etc.) pull in too many deps for what
    amounts to "connect to a WS, parse JSON, update a dict". This is 100 lines.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime

import websockets
from websockets.exceptions import ConnectionClosed

from src.newsalpha.models import PriceTick
from src.utils.logging import get_logger

logger = get_logger("newsalpha.coinbase_ws")

COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"


class CoinbaseTickerStream:
    """Streams live BTC/ETH/... spot prices from Coinbase's public ticker feed."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        on_tick: Callable[[PriceTick], None] | None = None,
    ):
        """Args:
            symbols: list of Coinbase product_ids, e.g. ["BTC-USD", "ETH-USD"]
            on_tick: optional callback invoked for every price update
        """
        self.symbols = symbols or ["BTC-USD"]
        self.on_tick = on_tick
        self._prices: dict[str, PriceTick] = {}
        self._ws = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def get_price(self, symbol: str = "BTC-USD") -> float | None:
        """Latest known price for a symbol. None until first tick arrives."""
        tick = self._prices.get(symbol)
        return tick.price if tick else None

    def get_tick(self, symbol: str = "BTC-USD") -> PriceTick | None:
        return self._prices.get(symbol)

    async def start(self) -> None:
        """Start the background reconnecting stream."""
        self._stop.clear()
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def wait_for_first_tick(self, symbol: str = "BTC-USD", timeout: float = 10.0) -> bool:
        """Block until a price is available or timeout. Returns True on success."""
        start = asyncio.get_event_loop().time()
        while not self._prices.get(symbol):
            if asyncio.get_event_loop().time() - start > timeout:
                return False
            await asyncio.sleep(0.05)
        return True

    async def _run_forever(self) -> None:
        """Outer loop: reconnect with backoff on disconnect."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_and_pump()
                backoff = 1.0  # reset on clean connect
            except ConnectionClosed as e:
                logger.warning("coinbase_ws_closed", code=e.code, reason=str(e.reason))
            except Exception as e:
                logger.error("coinbase_ws_error", error=str(e))

            if self._stop.is_set():
                break
            wait = min(backoff, 30.0)
            logger.info("coinbase_ws_reconnect_wait", seconds=wait)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=wait)
                break  # stop was set
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 30.0)

    async def _connect_and_pump(self) -> None:
        async with websockets.connect(
            COINBASE_WS_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": self.symbols,
                "channels": ["ticker"],
            }
            await ws.send(json.dumps(subscribe_msg))
            logger.info("coinbase_ws_connected", symbols=self.symbols)

            messages_seen = 0
            async for raw in ws:
                if self._stop.is_set():
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                # Log the first few messages at info level — crucial for debugging
                # subscription issues, auth errors, geo-restrictions, etc.
                messages_seen += 1
                if messages_seen <= 5:
                    logger.info(
                        "coinbase_ws_msg",
                        n=messages_seen,
                        type=msg.get("type"),
                        preview=str(msg)[:300],
                    )

                # Explicitly surface any error messages Coinbase sends back
                if msg.get("type") == "error":
                    logger.error("coinbase_ws_error_msg", message=msg.get("message"), reason=msg.get("reason"))
                    continue

                if msg.get("type") != "ticker":
                    continue

                symbol = msg.get("product_id")
                price_str = msg.get("price")
                if not symbol or not price_str:
                    continue
                try:
                    price = float(price_str)
                except ValueError:
                    continue

                tick = PriceTick(symbol=symbol, price=price, source="coinbase")
                self._prices[symbol] = tick
                if self.on_tick:
                    try:
                        self.on_tick(tick)
                    except Exception as e:
                        logger.warning("on_tick_callback_error", error=str(e))
