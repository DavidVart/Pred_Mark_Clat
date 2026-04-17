"""Stage 5: Trade Executor — place orders and manage positions."""

from __future__ import annotations

import uuid
from datetime import datetime

from src.clients.base_client import ExchangeClient
from src.db.manager import DatabaseManager
from src.models.portfolio import PortfolioState
from src.models.trade import Position, TradeExecution, TradeSignal
from src.utils.logging import get_logger

logger = get_logger("executor")


class TradeExecutor:
    """Executes trades and manages open positions."""

    def __init__(
        self,
        clients: dict[str, ExchangeClient],
        db: DatabaseManager,
        paper_mode: bool = True,
        slippage_limit: float = 0.02,
        stop_loss_pct: float = 0.10,
        take_profit_pct: float = 0.25,
        prefer_maker_orders: bool = True,
        maker_offset: float = 0.01,
    ):
        self.clients = clients
        self.db = db
        self.paper_mode = paper_mode
        self.slippage_limit = slippage_limit
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.prefer_maker_orders = prefer_maker_orders
        self.maker_offset = maker_offset

    async def execute(self, signal: TradeSignal) -> TradeExecution | None:
        """Execute a trade signal."""
        client = self.clients.get(signal.platform)
        if not client:
            logger.error("no_client", platform=signal.platform)
            return None

        # Check slippage — get current price
        current_market = await client.get_market(signal.market_id)
        if current_market:
            current_price = current_market.yes_price if signal.side == "yes" else current_market.no_price
            slippage = abs(current_price - signal.market_price)
            if slippage > self.slippage_limit:
                logger.warning(
                    "slippage_abort",
                    market=signal.title[:50],
                    expected=signal.market_price,
                    current=current_price,
                    slippage=slippage,
                )
                return None

        # Compute entry price — taker or maker
        # In maker mode we post a limit 1 cent below market price (in our favor).
        # On Polymarket international this gets 0% maker fee instead of 0.75-1.80% taker.
        # On Kalshi fee is per-contract regardless, but the limit captures the spread.
        # Paper-mode caveat: we assume the limit fills at the posted price. Real fill
        # probability < 100%, so paper PnL is slightly optimistic on maker runs.
        order_type = "maker" if self.prefer_maker_orders else "taker"
        if self.prefer_maker_orders:
            # Round to 2 decimals so Polymarket/Kalshi tick sizes are respected
            price = round(max(0.01, signal.market_price - self.maker_offset), 2)
        else:
            price = signal.market_price

        size = signal.dollar_size / price if price > 0 else 0

        if size <= 0:
            logger.warning("zero_size", market=signal.title[:50])
            return None

        try:
            order_id = await client.place_order(
                market_id=signal.market_id,
                side=signal.side,
                size=size,
                price=price,
            )
        except Exception as e:
            logger.error("order_failed", market=signal.title[:50], error=str(e))
            return None

        # Record execution
        execution = TradeExecution(
            execution_id=order_id,
            signal=signal,
            fill_price=price,
            quantity=size,
            total_cost=signal.dollar_size,
            is_paper=self.paper_mode,
            platform_order_id=order_id,
        )

        # Save position to DB
        position_id = f"{signal.platform}-{uuid.uuid4().hex[:8]}"
        await self.db.insert_position({
            "position_id": position_id,
            "market_id": signal.market_id,
            "platform": signal.platform,
            "title": signal.title,
            "side": signal.side,
            "entry_price": price,
            "quantity": size,
            "cost_basis": signal.dollar_size,
            "current_price": price,
            "stop_loss": self.stop_loss_pct,
            "take_profit": self.take_profit_pct,
            "is_paper": 1 if self.paper_mode else 0,
            "category": "",
            "opened_at": datetime.utcnow().isoformat(),
        })

        logger.info(
            "trade_executed",
            position_id=position_id,
            market=signal.title[:50],
            side=signal.side,
            order_type=order_type,
            price=price,
            market_price=signal.market_price,
            size=size,
            dollar=f"${signal.dollar_size:.2f}",
            paper=self.paper_mode,
        )

        return execution

    async def check_exits(self) -> int:
        """Check all open positions for stop-loss or take-profit triggers."""
        positions = await self.db.get_open_positions()
        exits = 0

        for pos_data in positions:
            client = self.clients.get(pos_data["platform"])
            if not client:
                continue

            # Get current market price
            market = await client.get_market(pos_data["market_id"])
            if not market:
                continue

            current_price = market.yes_price if pos_data["side"] == "yes" else market.no_price

            # Update current price in DB
            await self.db.update_position_price(pos_data["position_id"], current_price)

            # Calculate unrealized PnL
            entry = pos_data["entry_price"]
            quantity = pos_data["quantity"]
            # Same PnL formula for YES and NO — both are assets paying $1 at
            # resolution, and current_price tracks the side we hold.
            pnl = (current_price - entry) * quantity

            cost_basis = pos_data["cost_basis"]
            pnl_pct = pnl / cost_basis if cost_basis > 0 else 0

            # Check stop-loss
            stop_loss = pos_data.get("stop_loss", self.stop_loss_pct)
            if pnl_pct <= -stop_loss and stop_loss > 0:
                logger.info(
                    "stop_loss_triggered",
                    position=pos_data["position_id"],
                    pnl_pct=f"{pnl_pct:.1%}",
                )
                await self.db.close_position(
                    pos_data["position_id"],
                    exit_price=current_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    outcome="loss",
                    failure_class="stop_loss",
                )
                exits += 1
                continue

            # Check take-profit
            take_profit = pos_data.get("take_profit", self.take_profit_pct)
            if pnl_pct >= take_profit and take_profit > 0:
                logger.info(
                    "take_profit_triggered",
                    position=pos_data["position_id"],
                    pnl_pct=f"{pnl_pct:.1%}",
                )
                await self.db.close_position(
                    pos_data["position_id"],
                    exit_price=current_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    outcome="win",
                )
                exits += 1
                continue

            # Check if market is settled / closed
            if market.status in ("closed", "settled", "finalized"):
                # Market resolved — determine outcome.
                # final_price is the YES-side resolution value.
                final_price = 1.0 if market.yes_price > 0.95 else (0.0 if market.yes_price < 0.05 else current_price)
                if pos_data["side"] == "yes":
                    # YES share pays final_price at resolution
                    pnl = (final_price - entry) * quantity
                else:
                    # NO share pays (1 - final_price) at resolution
                    no_payout = 1.0 - final_price
                    pnl = (no_payout - entry) * quantity
                pnl_pct = pnl / cost_basis if cost_basis > 0 else 0

                outcome = "win" if pnl > 0 else "loss"
                logger.info(
                    "market_settled",
                    position=pos_data["position_id"],
                    outcome=outcome,
                    pnl=f"${pnl:.2f}",
                )
                await self.db.close_position(
                    pos_data["position_id"],
                    exit_price=final_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                )
                exits += 1

        if exits > 0:
            logger.info("exits_processed", count=exits)

        return exits

    async def get_portfolio_state(self, initial_capital: float = 1000.0) -> PortfolioState:
        """Build current portfolio state from DB."""
        positions_data = await self.db.get_open_positions()
        daily_pnl = await self.db.get_daily_pnl()
        daily_cost = await self.db.get_daily_ai_cost()

        positions = []
        total_exposure = 0.0

        for p in positions_data:
            pos = Position(
                position_id=p["position_id"],
                market_id=p["market_id"],
                platform=p["platform"],
                title=p["title"],
                side=p["side"],
                entry_price=p["entry_price"],
                quantity=p["quantity"],
                cost_basis=p["cost_basis"],
                current_price=p.get("current_price", p["entry_price"]),
                stop_loss=p.get("stop_loss", 0),
                take_profit=p.get("take_profit", 0),
                is_paper=bool(p.get("is_paper", 1)),
                category=p.get("category", ""),
            )
            positions.append(pos)
            total_exposure += pos.cost_basis

        # Calculate total value
        unrealized = sum(p.unrealized_pnl for p in positions)
        cash = initial_capital - total_exposure
        total_value = cash + total_exposure + unrealized

        return PortfolioState(
            cash=cash,
            positions=positions,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_cost=daily_cost,
            peak_value=max(total_value, initial_capital),
            is_paper=self.paper_mode,
        )
