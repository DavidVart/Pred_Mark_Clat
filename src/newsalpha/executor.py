"""NewsAlpha executor — fast paper/gray/live trading on divergence signals.

Three execution modes:
    - "paper": ideal fills at signal price, 0% fees — "best case" reference
    - "gray":  simulated friction via SlippageSimulator — realistic paper
    - "live":  real Polymarket orders (Day 2+, not in this module yet)

Running paper + gray in parallel on the same signals lets us measure how
much "edge" is real vs an artifact of optimistic paper fills.

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
from typing import Literal

from src.newsalpha.db import NewsAlphaDB
from src.newsalpha.models import DivergenceSignal, MarketQuote
from src.newsalpha.slippage import SlippageSimulator
from src.utils.logging import get_logger

logger = get_logger("newsalpha.executor")

ExecutionMode = Literal["paper", "gray", "live"]


class NewsAlphaExecutorConfig:
    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.15,
        max_position_pct: float = 0.05,
        # Stop-loss and trailing-activation must be WIDER than round-trip friction
        # (~5-8% on Polymarket) or every gray/live trade insta-stops on the
        # entry spread. These defaults are calibrated for post-friction trading.
        stop_loss_pct: float = 0.08,
        trailing_profit_lock_pct: float = 0.50,
        trailing_activation_pct: float = 0.04,
        flatten_before_resolution_seconds: float = 60.0,
        max_positions: int = 3,
        paper_mode: bool = True,
        # Live-mode-only safety caps (enforced in addition to the above)
        live_max_position_usd: float = 5.0,
        live_max_daily_loss_usd: float = 20.0,
        live_max_daily_opens: int = 30,
        # Sanity cap: refuse to open a position with size (shares) exceeding
        # this. Catches "buy 5000 shares at 1¢" pathology on deep-OTM markets.
        max_shares_per_position: float = 1000.0,
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
        self.live_max_position_usd = live_max_position_usd
        self.live_max_daily_loss_usd = live_max_daily_loss_usd
        self.live_max_daily_opens = live_max_daily_opens
        self.max_shares_per_position = max_shares_per_position


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
        # On prediction markets, YES and NO shares are BOTH assets that pay
        # $1 at resolution. For either side, PnL per share = exit - entry.
        # current_price is the market price of the SAME side we bought, so
        # a single formula works for both.
        return (self.current_price - self.entry_price) * self.size

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.cost_basis if self.cost_basis > 0 else 0.0

    @property
    def seconds_until_resolution(self) -> float:
        return max(0.0, (self.window_end - datetime.utcnow()).total_seconds())


class NewsAlphaExecutor:
    """Manages positions in one execution mode (paper/gray/live).

    Multiple executors can coexist on the same DB, tagging their positions
    and trades with their mode so paper vs gray vs live can be compared.
    """

    def __init__(
        self,
        config: NewsAlphaExecutorConfig,
        db: NewsAlphaDB,
        mode: ExecutionMode = "paper",
        slippage: SlippageSimulator | None = None,
    ):
        self.config = config
        self.db = db
        self.mode = mode
        # Slippage required for gray mode; ignored in paper; used for live too.
        if mode == "gray" and slippage is None:
            raise ValueError("gray mode requires a SlippageSimulator")
        self.slippage = slippage
        self._positions: dict[str, OpenPosition] = {}
        self._realized_pnl: float = 0.0
        self._trade_count: int = 0
        # Live-mode-only counters for daily caps
        self._daily_opens: int = 0
        self._daily_opens_date: str = datetime.utcnow().strftime("%Y-%m-%d")

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

    async def on_signal(self, signal: DivergenceSignal, quote: MarketQuote | None = None) -> bool:
        """Handle a new divergence signal. Returns True if a position was opened.

        In gray/live mode, passes the signal through the slippage simulator.
        If the simulator says "not filled" (adverse move, maker not lifted,
        etc.), the signal is consumed but no position is opened.
        """

        # Already have a position on this market?
        existing = self._positions.get(signal.market_id)
        if existing is not None:
            if existing.side != signal.side:
                await self._close_position(existing, signal.market_price, "side_reversal")
            else:
                return False

        # Check position limits
        if self.open_position_count >= self.config.max_positions:
            return False

        # Don't open too close to resolution
        if signal.seconds_remaining < self.config.flatten_before_resolution_seconds * 2:
            return False

        # Live-mode daily caps
        if self.mode == "live":
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if today != self._daily_opens_date:
                self._daily_opens = 0
                self._daily_opens_date = today
            if self._daily_opens >= self.config.live_max_daily_opens:
                logger.warning("live_daily_opens_cap_reached", cap=self.config.live_max_daily_opens)
                return False

        # Kelly-lite sizing
        dollar_size = self._compute_size(signal.edge)
        if dollar_size <= 0:
            return False
        # Live-mode dollar cap (MUCH tighter than paper)
        if self.mode == "live":
            dollar_size = min(dollar_size, self.config.live_max_position_usd)

        # --- Slippage simulation (gray mode) ---
        signal_price = signal.market_price
        entry_price = signal_price
        entry_fees = 0.0
        entry_latency_ms = 0
        slippage_bps = 0.0

        if self.mode == "gray" and self.slippage is not None:
            sim_quote = quote or self._quote_from_signal(signal)
            exec_result = self.slippage.simulate_entry(signal, sim_quote, order_type="taker")
            if not exec_result.filled:
                logger.info(
                    "gray_fill_rejected",
                    mode=self.mode,
                    market=signal.title[:40],
                    reason=exec_result.reason,
                    signal_price=signal_price,
                    quote_at_arrival=exec_result.quote_at_arrival_price,
                )
                return False
            entry_price = exec_result.fill_price
            # Compute size from slipped price; then apply fees
            size = dollar_size / entry_price if entry_price > 0 else 0
            if size <= 0:
                return False
            self.slippage.apply_fee(exec_result, size)
            entry_fees = exec_result.fees_paid
            entry_latency_ms = exec_result.latency_ms
            slippage_bps = exec_result.slippage_bps
            # Re-anchor cost_basis to include fees so PnL accounts for them
            dollar_size = exec_result.effective_cost
        else:
            size = dollar_size / entry_price if entry_price > 0 else 0
            if size <= 0:
                return False

        # Sanity: refuse enormous share counts. If size > max_shares, the
        # entry_price must be very low (< 5c) and we're on a thin/tick-pathology
        # market. The divergence detector's price filter should have caught this
        # but belt-and-suspenders.
        if size > self.config.max_shares_per_position:
            logger.warning(
                "refused_oversized_position",
                mode=self.mode,
                market=signal.title[:40],
                size=round(size, 0),
                entry_price=entry_price,
                cap=self.config.max_shares_per_position,
            )
            return False

        pos = OpenPosition(
            position_id=f"na-{self.mode}-{uuid.uuid4().hex[:10]}",
            market_id=signal.market_id,
            title=signal.title,
            side=signal.side,
            entry_price=entry_price,
            size=size,
            cost_basis=dollar_size,
            window_end=datetime.utcnow(),
            signal_edge=signal.edge,
        )
        from datetime import timedelta
        pos.window_end = datetime.utcnow() + timedelta(seconds=signal.seconds_remaining)

        # Stash slippage metadata for persistence on close
        pos._signal_price = signal_price
        pos._entry_fees = entry_fees
        pos._entry_latency_ms = entry_latency_ms
        pos._entry_slippage_bps = slippage_bps

        self._positions[signal.market_id] = pos
        if self.mode == "live":
            self._daily_opens += 1

        # Persist to DB
        await self.db.db.execute(
            """INSERT INTO na_positions (position_id, market_id, title, side,
               entry_price, size, cost_basis, window_end, signal_edge, is_paper,
               execution_mode, signal_price, entry_fees, entry_latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pos.position_id, pos.market_id, pos.title, pos.side,
                pos.entry_price, pos.size, pos.cost_basis,
                pos.window_end.isoformat(), pos.signal_edge,
                1 if self.mode != "live" else 0,
                self.mode, signal_price, entry_fees, entry_latency_ms,
            ),
        )
        await self.db.db.commit()

        logger.info(
            "position_opened",
            mode=self.mode,
            position_id=pos.position_id,
            market=pos.title[:50],
            side=pos.side,
            signal_price=round(signal_price, 3),
            entry_price=round(pos.entry_price, 3),
            slippage_bps=round(slippage_bps, 1),
            fees=f"${entry_fees:.3f}",
            size=round(pos.size, 2),
            cost=f"${pos.cost_basis:.2f}",
            edge=round(pos.signal_edge, 3),
        )
        return True

    @staticmethod
    def _quote_from_signal(signal: DivergenceSignal) -> MarketQuote:
        """Synthesize a minimal MarketQuote from a signal for slippage sim.

        Only used when we haven't been passed a fresh quote. The slippage sim
        cares about the current market price (from signal) and timing (from
        seconds_remaining) — not the full quote.
        """
        from datetime import timedelta
        now = datetime.utcnow()
        return MarketQuote(
            market_id=signal.market_id,
            title=signal.title,
            yes_price=signal.market_price if signal.side == "yes" else 1 - signal.market_price,
            no_price=1 - signal.market_price if signal.side == "yes" else signal.market_price,
            window_start=now - timedelta(seconds=60),
            window_end=now + timedelta(seconds=signal.seconds_remaining),
        )

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
        """Close a position and record the trade.

        In gray/live mode, exit price is slipped + exit fees deducted. Paper
        mode closes at the provided exit_price with no friction.
        """
        # Apply exit slippage + fees (gray/live modes)
        fees_paid_on_exit = 0.0
        if self.mode == "gray" and self.slippage is not None:
            slipped_price, fee_pct = self.slippage.simulate_exit(exit_price, pos.side, order_type="taker")
            fees_paid_on_exit = slipped_price * pos.size * fee_pct
            exit_price = slipped_price
            # Re-derive PnL at the slipped exit price
            pos.current_price = exit_price
            pnl = pos.pnl - fees_paid_on_exit - getattr(pos, "_entry_fees", 0.0)
        else:
            pnl = pos.pnl

        cost_basis = pos.cost_basis if pos.cost_basis > 0 else 1e-9
        pnl_pct = pnl / cost_basis
        hold_seconds = (datetime.utcnow() - pos.opened_at).total_seconds()
        outcome = "win" if pnl > 0 else "loss"

        self._realized_pnl += pnl
        self._trade_count += 1

        total_fees = getattr(pos, "_entry_fees", 0.0) + fees_paid_on_exit

        trade_id = f"nat-{self.mode}-{uuid.uuid4().hex[:10]}"
        await self.db.db.execute(
            """INSERT INTO na_trades (trade_id, market_id, title, side,
               entry_price, exit_price, size, pnl, pnl_pct, hold_seconds,
               outcome, exit_reason, signal_edge, is_paper, opened_at,
               execution_mode, signal_price, fees_paid, slippage_bps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id, pos.market_id, pos.title, pos.side,
                pos.entry_price, exit_price, pos.size, pnl, pnl_pct,
                hold_seconds, outcome, reason, pos.signal_edge,
                1 if self.mode != "live" else 0,
                pos.opened_at.isoformat(),
                self.mode,
                getattr(pos, "_signal_price", pos.entry_price),
                total_fees,
                getattr(pos, "_entry_slippage_bps", 0.0),
            ),
        )
        await self.db.db.execute(
            "DELETE FROM na_positions WHERE position_id = ?", (pos.position_id,)
        )
        await self.db.db.commit()

        self._positions.pop(pos.market_id, None)

        logger.info(
            "position_closed",
            mode=self.mode,
            position_id=pos.position_id,
            market=pos.title[:50],
            side=pos.side,
            entry=round(pos.entry_price, 3),
            exit=round(exit_price, 3),
            pnl=f"${pnl:+.2f}",
            fees=f"${total_fees:.3f}",
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
