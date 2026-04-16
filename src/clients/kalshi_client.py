"""Kalshi client — REST API with RSA-PSS authentication."""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.models.market import OrderBook, UnifiedMarket
from src.models.trade import Position
from src.utils.logging import get_logger

logger = get_logger("kalshi")


class KalshiClient:
    """Client for Kalshi prediction market exchange."""

    PROD_URL = "https://trading-api.kalshi.com"
    DEMO_URL = "https://demo-api.kalshi.co"

    def __init__(
        self,
        api_key: str = "",
        private_key_path: str = "kalshi_private_key.pem",
        use_demo: bool = True,
        live_mode: bool = False,
    ):
        self.api_key = api_key
        self.private_key_path = private_key_path
        self.live_mode = live_mode
        self.base_url = self.DEMO_URL if use_demo else self.PROD_URL
        self._private_key: rsa.RSAPrivateKey | None = None
        self._http = httpx.AsyncClient(timeout=30.0)
        self._last_request_time = 0.0
        self._min_request_interval = 0.2  # 200ms = 5 req/s

    @property
    def platform_name(self) -> str:
        return "kalshi"

    def _load_private_key(self) -> rsa.RSAPrivateKey:
        """Load RSA private key from PEM file."""
        if self._private_key is not None:
            return self._private_key

        key_path = Path(self.private_key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"Kalshi private key not found: {key_path}")

        key_data = key_path.read_bytes()
        self._private_key = serialization.load_pem_private_key(key_data, password=None)
        return self._private_key

    def _sign_request(self, method: str, path: str) -> dict[str, str]:
        """Generate RSA-PSS signature headers for Kalshi API."""
        timestamp_ms = str(int(time.time() * 1000))
        message = timestamp_ms + method.upper() + path
        message_bytes = message.encode("utf-8")

        private_key = self._load_private_key()
        signature = private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "Content-Type": "application/json",
        }

    async def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict | None = None,
        params: dict | None = None,
        retries: int = 3,
    ) -> dict:
        """Make an authenticated request with retry logic."""
        import asyncio

        await self._rate_limit()
        url = f"{self.base_url}{path}"
        headers = self._sign_request(method, path)

        for attempt in range(retries):
            try:
                resp = await self._http.request(
                    method, url, headers=headers, json=json_data, params=params
                )

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("rate_limited", attempt=attempt, wait=wait)
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning("server_error", status=resp.status_code, attempt=attempt)
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json() if resp.content else {}

            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning("request_retry", error=str(e), attempt=attempt)
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Request failed after {retries} retries: {method} {path}")

    async def get_active_markets(self, limit: int = 100) -> list[UnifiedMarket]:
        """Fetch active markets from Kalshi."""
        markets: list[UnifiedMarket] = []
        cursor: str | None = None

        while len(markets) < limit:
            params: dict = {"limit": min(limit - len(markets), 100), "status": "open"}
            if cursor:
                params["cursor"] = cursor

            try:
                data = await self._request("GET", "/trade-api/v2/markets", params=params)
            except Exception as e:
                logger.error("markets_fetch_failed", error=str(e))
                break

            for m in data.get("markets", []):
                try:
                    market = self._parse_market(m)
                    if market:
                        markets.append(market)
                except Exception as e:
                    logger.debug("market_parse_error", error=str(e))

            cursor = data.get("cursor")
            if not cursor or not data.get("markets"):
                break

        return markets[:limit]

    async def get_market(self, market_id: str) -> UnifiedMarket | None:
        """Fetch a single market by ticker."""
        try:
            data = await self._request("GET", f"/trade-api/v2/markets/{market_id}")
            market_data = data.get("market", data)
            return self._parse_market(market_data)
        except Exception as e:
            logger.error("market_fetch_failed", market_id=market_id, error=str(e))
            return None

    async def get_orderbook(self, market_id: str) -> OrderBook:
        """Get orderbook for a market."""
        try:
            data = await self._request("GET", f"/trade-api/v2/markets/{market_id}/orderbook")
            bids = [(float(o[0]) / 100, float(o[1])) for o in data.get("yes", [])]
            asks = [(float(o[0]) / 100, float(o[1])) for o in data.get("no", [])]
            return OrderBook(market_id=market_id, bids=bids, asks=asks)
        except Exception as e:
            logger.error("orderbook_failed", market_id=market_id, error=str(e))
            return OrderBook(market_id=market_id)

    async def place_order(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float,
    ) -> str:
        """Place a limit order on Kalshi."""
        if not self.live_mode:
            order_id = f"paper-kalshi-{uuid.uuid4().hex[:12]}"
            logger.info("paper_order", order_id=order_id, side=side, size=size, price=price)
            return order_id

        # Convert price from 0-1 to cents (1-99)
        price_cents = max(1, min(99, int(price * 100)))
        count = max(1, int(size))

        payload = {
            "ticker": market_id,
            "action": "buy",
            "side": side,
            "type": "limit",
            "count": count,
        }

        if side == "yes":
            payload["yes_price"] = price_cents
        else:
            payload["no_price"] = price_cents

        try:
            data = await self._request("POST", "/trade-api/v2/portfolio/orders", json_data=payload)
            order_id = data.get("order", {}).get("order_id", "unknown")
            logger.info("order_placed", order_id=order_id, ticker=market_id, side=side)
            return str(order_id)
        except Exception as e:
            logger.error("order_failed", error=str(e), ticker=market_id)
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.live_mode:
            return True
        try:
            await self._request("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
            return True
        except Exception as e:
            logger.error("cancel_failed", order_id=order_id, error=str(e))
            return False

    async def get_balance(self) -> float:
        """Get account balance in dollars."""
        try:
            data = await self._request("GET", "/trade-api/v2/portfolio/balance")
            # Balance is in cents
            return float(data.get("balance", 0)) / 100
        except Exception as e:
            logger.error("balance_failed", error=str(e))
            return 0.0

    async def get_positions(self) -> list[Position]:
        """Get open positions from Kalshi."""
        try:
            data = await self._request("GET", "/trade-api/v2/portfolio/positions")
            positions = []
            for p in data.get("market_positions", []):
                if p.get("position", 0) != 0:
                    pos = Position(
                        position_id=f"kalshi-{p.get('ticker', '')}",
                        market_id=p.get("ticker", ""),
                        platform="kalshi",
                        title=p.get("ticker", ""),
                        side="yes" if p.get("position", 0) > 0 else "no",
                        entry_price=float(p.get("average_price", 0)) / 100,
                        quantity=abs(p.get("position", 0)),
                        cost_basis=float(p.get("total_cost", 0)) / 100,
                        current_price=float(p.get("market_price", 0)) / 100,
                    )
                    positions.append(pos)
            return positions
        except Exception as e:
            logger.error("positions_failed", error=str(e))
            return []

    async def close(self) -> None:
        await self._http.aclose()

    def _parse_market(self, data: dict) -> UnifiedMarket | None:
        """Convert Kalshi market data to UnifiedMarket."""
        ticker = data.get("ticker", "")
        if not ticker:
            return None

        # Prices in Kalshi are in cents (1-99), convert to 0-1
        yes_price = float(data.get("yes_price", 50)) / 100
        no_price = float(data.get("no_price", 50)) / 100

        # If yes_price is already in 0-1 range (some API responses)
        if data.get("yes_price", 0) <= 1.0:
            yes_price = float(data.get("yes_price", 0.5))
            no_price = float(data.get("no_price", 0.5))

        # Parse expiration
        expiration = None
        exp_str = data.get("expiration_time") or data.get("close_time")
        if exp_str:
            try:
                expiration = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        volume = int(data.get("volume", 0) or 0)

        return UnifiedMarket(
            platform="kalshi",
            market_id=ticker,
            title=data.get("title", ticker),
            description=data.get("subtitle", ""),
            category=data.get("category", ""),
            yes_price=max(0.0, min(1.0, yes_price)),
            no_price=max(0.0, min(1.0, no_price)),
            volume=volume,
            expiration=expiration,
            status=data.get("status", "open"),
            outcomes=["Yes", "No"],
            url=f"https://kalshi.com/markets/{ticker}",
            ticker=ticker,
        )
