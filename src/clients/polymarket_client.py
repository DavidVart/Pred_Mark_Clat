"""Polymarket client — Gamma API for discovery, CLOB API for trading."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import httpx

from src.models.market import OrderBook, UnifiedMarket
from src.models.trade import Position
from src.utils.logging import get_logger

logger = get_logger("polymarket")


class PolymarketClient:
    """Client for Polymarket using Gamma API (discovery) and CLOB API (trading)."""

    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"

    def __init__(
        self,
        wallet_private_key: str = "",
        live_mode: bool = False,
    ):
        self.wallet_private_key = wallet_private_key
        self.live_mode = live_mode
        self._http = httpx.AsyncClient(timeout=30.0)
        self._clob_client = None

    @property
    def platform_name(self) -> str:
        return "polymarket"

    async def _init_clob(self) -> None:
        """Initialize the CLOB client for trading (lazy, only when needed)."""
        if self._clob_client is not None or not self.wallet_private_key:
            return
        try:
            from py_clob_client.client import ClobClient

            self._clob_client = ClobClient(
                self.CLOB_URL,
                key=self.wallet_private_key,
                chain_id=137,
            )
            creds = self._clob_client.create_or_derive_api_creds()
            self._clob_client.set_api_creds(creds)
            logger.info("clob_client_initialized")
        except Exception as e:
            logger.error("clob_init_failed", error=str(e))

    async def get_active_markets(self, limit: int = 100) -> list[UnifiedMarket]:
        """Fetch active markets from Gamma API."""
        markets: list[UnifiedMarket] = []
        offset = 0
        page_size = min(limit, 100)

        while len(markets) < limit:
            try:
                resp = await self._http.get(
                    f"{self.GAMMA_URL}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "archived": "false",
                        "limit": page_size,
                        "offset": offset,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error("gamma_fetch_failed", error=str(e), offset=offset)
                break

            if not data:
                break

            for m in data:
                try:
                    market = self._parse_market(m)
                    if market:
                        markets.append(market)
                except Exception as e:
                    logger.debug("market_parse_error", error=str(e), market_id=m.get("id"))

            offset += page_size
            if len(data) < page_size:
                break

        return markets[:limit]

    async def get_market(self, market_id: str) -> UnifiedMarket | None:
        """Fetch a single market by condition ID."""
        try:
            resp = await self._http.get(f"{self.GAMMA_URL}/markets/{market_id}")
            resp.raise_for_status()
            data = resp.json()
            return self._parse_market(data)
        except Exception as e:
            logger.error("market_fetch_failed", market_id=market_id, error=str(e))
            return None

    async def get_orderbook(self, market_id: str) -> OrderBook:
        """Get orderbook from CLOB API."""
        await self._init_clob()
        if self._clob_client:
            try:
                book = self._clob_client.get_order_book(market_id)
                bids = [(float(o.get("price", 0)), float(o.get("size", 0))) for o in book.get("bids", [])]
                asks = [(float(o.get("price", 0)), float(o.get("size", 0))) for o in book.get("asks", [])]
                return OrderBook(market_id=market_id, bids=bids, asks=asks)
            except Exception as e:
                logger.error("orderbook_fetch_failed", market_id=market_id, error=str(e))

        return OrderBook(market_id=market_id)

    async def place_order(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float,
    ) -> str:
        """Place a limit order on Polymarket CLOB."""
        if not self.live_mode:
            order_id = f"paper-poly-{uuid.uuid4().hex[:12]}"
            logger.info("paper_order", order_id=order_id, side=side, size=size, price=price)
            return order_id

        await self._init_clob()
        if not self._clob_client:
            raise RuntimeError("CLOB client not initialized — missing wallet key")

        try:
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side == "yes" else SELL
            order = self._clob_client.create_and_post_order({
                "token_id": market_id,
                "price": price,
                "size": size,
                "side": order_side,
            })
            order_id = order.get("orderID", order.get("id", "unknown"))
            logger.info("order_placed", order_id=order_id, side=side, size=size, price=price)
            return str(order_id)
        except Exception as e:
            logger.error("order_failed", error=str(e), side=side, size=size)
            raise

    async def cancel_order(self, order_id: str) -> bool:
        if not self.live_mode or not self._clob_client:
            return True
        try:
            self._clob_client.cancel(order_id)
            return True
        except Exception as e:
            logger.error("cancel_failed", order_id=order_id, error=str(e))
            return False

    async def get_balance(self) -> float:
        """Get USDC balance on Polygon (simplified)."""
        # In a full implementation, query on-chain USDC balance
        logger.warning("balance_check_not_implemented")
        return 0.0

    async def get_positions(self) -> list[Position]:
        """Get open positions (requires on-chain query)."""
        logger.warning("positions_check_not_implemented")
        return []

    async def close(self) -> None:
        await self._http.aclose()

    def _parse_market(self, data: dict) -> UnifiedMarket | None:
        """Convert Gamma API market data to UnifiedMarket."""
        if not data.get("question") and not data.get("title"):
            return None

        # Parse outcome prices
        yes_price = 0.5
        no_price = 0.5
        outcome_prices = data.get("outcomePrices")
        if outcome_prices:
            if isinstance(outcome_prices, str):
                try:
                    prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    prices = []
            else:
                prices = outcome_prices

            if len(prices) >= 2:
                yes_price = float(prices[0])
                no_price = float(prices[1])

        # Parse CLOB token IDs
        clob_ids = data.get("clobTokenIds")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except (json.JSONDecodeError, TypeError):
                clob_ids = None

        # Parse expiration
        expiration = None
        end_date = data.get("endDate") or data.get("end_date_iso")
        if end_date:
            try:
                expiration = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Parse volume
        volume = 0
        vol_raw = data.get("volume") or data.get("volumeNum") or 0
        try:
            volume = int(float(vol_raw))
        except (ValueError, TypeError):
            pass

        return UnifiedMarket(
            platform="polymarket",
            market_id=str(data.get("conditionId") or data.get("id", "")),
            title=data.get("question") or data.get("title", ""),
            description=data.get("description", ""),
            category=data.get("category", "") or data.get("groupItemTitle", ""),
            yes_price=max(0.0, min(1.0, yes_price)),
            no_price=max(0.0, min(1.0, no_price)),
            volume=volume,
            liquidity=float(data.get("liquidity", 0) or 0),
            expiration=expiration,
            status="active" if data.get("active") else "closed",
            outcomes=data.get("outcomes", ["Yes", "No"]) if isinstance(data.get("outcomes"), list) else ["Yes", "No"],
            url=f"https://polymarket.com/event/{data.get('slug', '')}",
            clob_token_ids=clob_ids,
            condition_id=data.get("conditionId"),
        )
