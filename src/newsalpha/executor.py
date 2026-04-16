"""NewsAlpha executor — fast paper trading on divergence signals.

Design goals:
    - Sub-second execution (no LLM in the loop — pure math)
    - Hard time-stop: flatten N seconds before market resolution
    - Rolling profit lock: once in profit ≥ X%, set trailing stop at 40% of peak
    - Fixed stop-loss: close if PnL drops below -Y%
    - Per-market position limit: max 1 open position per market
    - Kelly-lite sizing: position size = fraction_kelly * edge * bankroll
    - Mandatory paper-mode for v1

Exit hierarchy (first match wins):
    1. Time stop — flatten before resolution (no gap risk)
    2. Stop-loss — cut losers fast
    3. Trailing profit lock — let winners run, lock in gains
    4. Signal reversal — if our side flips (was YES, now NO), close and reverse

Position lifecycle:
    signal → open → {monitor on each cycle} → exit → record trade
"""

from __future__ import annotations

import uuid
from datetime import datetime

from src.newsalpha.db import NewsAlphaDB
from src.newsalpha.models import DivergenceSignal, MarketQuote
from src.utils.logging import get_logger

logger = get_logger("newsalpha.executor")


class NewsAlphaExecutorConfig:
    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.15,      # more aggressive than LLM bot's 0.25 because trades are shorter
        max_position_pct: float = 0.05,     # max 5% of bankroll per trade
        stop_loss_pct: float = 0.03,        # -3% stop (tight for short-duration)
        trailing_profit_lock_pct: float = 0.40,  # lock in 40% of peak unrealized gain
        trailing_activation_pct: float = 0.02,   # start trailing after +2% gain
        flatten_before_resolution_seconds: float = 60.0,  # close 60s before window ends
        max_positions: int = 3,             # max simultaneous positions
        paper_mode: bool = True,
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_profit_lock_pct = trailing_profit_lock_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.flatten_before_resolution_seconds = flatten_before_resolution_seconds
        self.max_positions = max_positions
        self.paper_mode = paper_mode


class OpenPosition:
    """In-memory tracked position (also persisted to DB)."""

    def __init__(
        self,
        position_id: str,
        market_id: str,
        title: str,
        side: str,
        entry_price: float,
        size: float,
        cost_basis: float,
        window_end: datetime,
        signal_edge: float,
    ):
        self.position_id = position_id
        self.market_id = market_id
        self.title = title
        self.side = side
        self.entry_price = entry_price
        self.size = size
        self.cost_basis = cost_basis
        self.window_end = window_end
        self.signal_edge = signal_edge
        self.opened_at = datetime.utcnow()
        self.peak_pnl_pct: float = 0.0
        self.current_price: float = entry_price

    def update_price(self, market_price: float) -> None:
        self.current_price = market_price
        pnl_pct = self.pnl_pct
        if pnl_pct > self.peak_pnl_pct:
            self.peak_pnl_pct = pnl_pct

    @property
    def pnl(self) -> float:
        if self.side == "yes":
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.cost_basis if self.cost_basis > 0 else 0.0

    @property
    def seconds_until_resolution(self) -> float:
        return max(0.0, (self.window_end - datetime.utcnow()).total_seconds())


class NewsAlphaExecutor:
    """Manages fast paper-trade positions based on divergence signals."""

    def __init__(self, config: NewsAlphaExecutorConfig, db: NewsAlphaDB):
        self.config = config
        self.db = db
        self._positions: dict[str, OpenPosition] = {}  # market_id → position
        self._realized_pnl: float = 0.0
        self._trade_count: int = 0

    @property
    def open_position_count(self) -> int:
        return len(self._positions)

    @property
    def current_bankroll(self) -> float:
        # Bankroll = initial + realized + unrealized
        unrealized = sum(p.pnl for p in self._positions.values())
        exposure = sum(p.cost_basis for p in self._positions.values())
        return self.config.bankroll + self._realized_pnl - exposure + sum(
            p.cost_basis + p.pnl for p in self._positions.values()
        )

    def has_position(self, market_id: str) -> bool:
        return market_id in self._positions

    async def on_signal(self, signal: DivergenceSignal) -> bool:
        """Handle a new divergence signal. Returns True if a position was opened."""

        # Already have a position on this market?
        existing = self._positions.get(signal.market_id)
        if existing is not None:
            if existing.side != signal.side:
                # Side flipped — close existing, then open new
                await self._close_position(existing, signal.market_price, "side_reversal")
            else:
                return False  # Same side, already in — skip

        # Check position limits
        if self.open_position_count >= self.config.max_positions:
            return False

        # Don't open too close to resolution
        if signal.seconds_remaining < self.config.flatten_before_resolution_seconds * 2:
            return False

        # Kelly-lite sizing
        dollar_size = self._compute_size(signal.edge)
        if dollar_size <= 0:
            return False

        price = signal.market_price
        size = dollar_size / price if price > 0 else 0
        if size <= 0:
            return False

        pos = OpenPosition(
            position_id=f"na-{uuid.uuid4().hex[:10]}",
            market_id=signal.market_id,
            title=signal.title,
            side=signal.side,
            entry_price=price,
            size=size,
            cost_basis=dollar_size,
            window_end=datetime.utcnow(),  # Will be updated when we have quote
            signal_edge=signal.edge,
        )
        # Approximate window_end from signal's seconds_remaining
        from datetime import timedelta
        pos.window_end = datetime.utcnow() + timedelta(seconds=signal.seconds_remaining)

        self._positions[signal.market_id] = pos

        # Persist to DB
        await self.db.db.execute(
            """INSERT INTO na_positions (position_id, market_id, title, side,
               entry_price, size, cost_basis, window_end, signal_edge, is_paper)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pos.position_id, pos.market_id, pos.title, pos.side,
                pos.entry_price, pos.size, pos.cost_basis,
                pos.window_end.isoformat(), pos.signal_edge,
                1 if self.config.paper_mode else 0,
            ),
        )
        await self.db.db.commit()

        logger.info(
            "position_opened",
            position_id=pos.position_id,
            market=pos.title[:50],
            side=pos.side,
            price=round(pos.entry_price, 3),
            size=round(pos.size, 2),
            cost=f"${pos.cost_basis:.2f}",
            edge=round(pos.signal_edge, 3),
            paper=self.config.paper_mode,
        )
        return True

    async def check_exits(self, quotes: dict[str, MarketQuote]) -> int:
        """Check all open positions for exit conditions. Call every cycle.

        Args:
            quotes: dict of market_id → latest MarketQuote

        Returns:
            Number of positions closed this cycle.
        """
        exits = 0
        to_close: list[tuple[OpenPosition, float, str]] = []

        for market_id, pos in list(self._positions.items()):
            quote = quotes.get(market_id)
            if quote is None:
                continue

            # Update current market price and resolution time from quote
            current_price = quote.yes_price if pos.side == "yes" else quote.no_price
            pos.update_price(current_price)
            # Quote has the authoritative resolution time; update position
            pos.window_end = quote.window_end

            # Exit check 1: Time stop — flatten before resolution
            if pos.seconds_until_resolution <= self.config.flatten_before_resolution_seconds:
                to_close.append((pos, current_price, "time_stop"))
                continue

            # Exit check 2: Stop-loss
            if pos.pnl_pct <= -self.config.stop_loss_pct:
                to_close.append((pos, current_price, "stop_loss"))
                continue

            # Exit check 3: Trailing profit lock
            if pos.peak_pnl_pct >= self.config.trailing_activation_pct:
                # Trailing stop level = peak PnL * (1 - lock_pct)
                trail_level = pos.peak_pnl_pct * (1.0 - self.config.trailing_profit_lock_pct)
                if pos.pnl_pct <= trail_level:
                    to_close.append((pos, current_price, "trailing_lock"))
                    continue

        for pos, exit_price, reason in to_close:
            await self._close_position(pos, exit_price, reason)
            exits += 1

        return exits

    async def _close_position(self, pos: OpenPosition, exit_price: float, reason: str) -> None:
        """Close a position and record the trade."""
        pnl = pos.pnl
        pnl_pct = pos.pnl_pct
        hold_seconds = (datetime.utcnow() - pos.opened_at).total_seconds()
        outcome = "win" if pnl > 0 else "loss"

        self._realized_pnl += pnl
        self._trade_count += 1

        trade_id = f"nat-{uuid.uuid4().hex[:10]}"
        await self.db.db.execute(
            """INSERT INTO na_trades (trade_id, market_id, title, side,
               entry_price, exit_price, size, pnl, pnl_pct, hold_seconds,
               outcome, exit_reason, signal_edge, is_paper, opened_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id, pos.market_id, pos.title, pos.side,
                pos.entry_price, exit_price, pos.size, pnl, pnl_pct,
                hold_seconds, outcome, reason, pos.signal_edge,
                1 if self.config.paper_mode else 0,
                pos.opened_at.isoformat(),
            ),
        )
        # Remove from positions table
        await self.db.db.execute(
            "DELETE FROM na_positions WHERE position_id = ?", (pos.position_id,)
        )
        await self.db.db.commit()

        # Remove from in-memory tracker
        self._positions.pop(pos.market_id, None)

        logger.info(
            "position_closed",
            position_id=pos.position_id,
            market=pos.title[:50],
            side=pos.side,
            entry=round(pos.entry_price, 3),
            exit=round(exit_price, 3),
            pnl=f"${pnl:+.2f}",
            pnl_pct=f"{pnl_pct:+.1%}",
            hold=f"{hold_seconds:.0f}s",
            reason=reason,
            outcome=outcome,
        )

    def _compute_size(self, edge: float) -> float:
        """Kelly-lite position sizing. edge is a probability differential (0-1)."""
        # Raw Kelly: f* = edge / odds. For binary at even odds, f* ≈ 2 * edge.
        # We use a fractional multiplier to be conservative.
        kelly = 2.0 * edge * self.config.kelly_fraction
        max_pct = self.config.max_position_pct
        sized = min(kelly, max_pct)
        return max(0.0, sized * self.current_bankroll)

    def get_summary(self) -> dict:
        """Summary stats for CLI display."""
        positions = list(self._positions.values())
        total_unrealized = sum(p.pnl for p in positions)
        return {
            "open_positions": len(positions),
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": total_unrealized,
            "total_pnl": self._realized_pnl + total_unrealized,
            "trade_count": self._trade_count,
            "bankroll": self.current_bankroll,
            "positions": [
                {
                    "id": p.position_id,
                    "market": p.title[:50],
                    "side": p.side,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl": p.pnl,
                    "pnl_pct": p.pnl_pct,
                    "hold_seconds": (datetime.utcnow() - p.opened_at).total_seconds(),
                    "seconds_to_resolution": p.seconds_until_resolution,
                }
                for p in positions
            ],
        }
