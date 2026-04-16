"""Stage 4: Risk Manager — deterministic risk checks and Kelly sizing. NO LLM calls."""

from __future__ import annotations

import math

from config.settings import RiskConfig
from src.db.manager import DatabaseManager
from src.models.portfolio import PortfolioState
from src.models.prediction import EnsembleResult
from src.models.trade import TradeSignal
from src.models.market import UnifiedMarket
from src.pipeline.market_filters import classify_market, cluster_key_from_title
from src.utils import kill_switch
from src.utils.logging import get_logger

logger = get_logger("risk_manager")


class RiskManager:
    """
    Deterministic risk validation layer. ALL checks are pure Python math.
    Every check must pass before a trade can execute.
    """

    def __init__(self, config: RiskConfig, db: DatabaseManager):
        self.config = config
        self.db = db

    async def approve_trade(
        self,
        market: UnifiedMarket,
        prediction: EnsembleResult,
        portfolio: PortfolioState,
    ) -> tuple[TradeSignal | None, str]:
        """
        Run all risk checks. Returns (TradeSignal, "") if approved,
        or (None, rejection_reason) if rejected.
        """
        checks = [
            self._check_kill_switch,
            lambda m, p, pf: self._check_drawdown(pf),
            lambda m, p, pf: self._check_daily_loss(pf),
            lambda m, p, pf: self._check_position_count(pf),
            lambda m, p, pf: self._check_edge(p),
            lambda m, p, pf: self._check_confidence(p),
            lambda m, p, pf: self._check_concentration(m, pf),
            lambda m, p, pf: self._check_cluster_limit(m, pf),
        ]

        for check in checks:
            ok, reason = await check(market, prediction, portfolio)
            if not ok:
                logger.info("trade_rejected", reason=reason, market=market.title[:50])
                return None, reason

        # All checks passed — calculate position size
        signal = self._build_signal(market, prediction, portfolio)

        logger.info(
            "trade_approved",
            market=market.title[:50],
            side=signal.side,
            edge=f"{signal.edge:+.3f}",
            kelly=f"{signal.kelly_size:.4f}",
            dollar_size=f"${signal.dollar_size:.2f}",
        )

        return signal, ""

    def _build_signal(
        self,
        market: UnifiedMarket,
        prediction: EnsembleResult,
        portfolio: PortfolioState,
    ) -> TradeSignal:
        """Build a sized trade signal."""
        edge = prediction.edge
        side = prediction.trade_side
        confidence = prediction.final_confidence

        # Determine odds for Kelly
        if side == "yes":
            market_price = market.yes_price
            win_prob = prediction.weighted_probability
        else:
            market_price = market.no_price
            win_prob = 1.0 - prediction.weighted_probability

        # Kelly sizing
        kelly_fraction = self._kelly_criterion(win_prob, market_price, confidence)

        # Dollar size
        dollar_size = kelly_fraction * portfolio.total_value

        return TradeSignal(
            market_id=market.market_id,
            platform=market.platform,
            title=market.title,
            side=side,
            predicted_probability=prediction.weighted_probability,
            market_price=market.yes_price,
            edge=edge,
            confidence=confidence,
            kelly_size=kelly_fraction,
            dollar_size=dollar_size,
        )

    def _kelly_criterion(self, win_prob: float, price: float, confidence: float) -> float:
        """
        Fractional Kelly Criterion for position sizing.

        f* = (b*p - q) / b  where b = net odds, p = win prob, q = 1-p
        Then multiply by kelly_fraction (0.25 = quarter-Kelly)
        And scale by confidence.
        """
        if price <= 0 or price >= 1:
            return 0.0

        # Net odds: if you buy at price P and win, you get (1-P)/P net return
        b = (1.0 - price) / price
        if b <= 0:
            return 0.0

        p = win_prob
        q = 1.0 - p

        kelly = (b * p - q) / b

        if kelly <= 0:
            return 0.0

        # Apply fractional Kelly
        sized = kelly * self.config.kelly_fraction

        # Scale by confidence
        sized *= confidence

        # Cap at max position size
        sized = min(sized, self.config.max_position_size_pct)

        return max(0.0, sized)

    # --- Individual Risk Checks ---

    async def _check_kill_switch(self, market, prediction, portfolio) -> tuple[bool, str]:
        halted, reason = await kill_switch.is_halted(self.db)
        if halted:
            return False, f"Kill switch active: {reason}"
        return True, ""

    async def _check_drawdown(self, portfolio: PortfolioState) -> tuple[bool, str]:
        if portfolio.current_drawdown >= self.config.max_drawdown_pct:
            # Auto-activate kill switch on drawdown breach
            await kill_switch.activate(
                self.db,
                f"Max drawdown breached: {portfolio.current_drawdown:.1%} >= {self.config.max_drawdown_pct:.1%}",
                "risk_manager",
            )
            return False, f"Max drawdown {portfolio.current_drawdown:.1%} >= limit {self.config.max_drawdown_pct:.1%}"
        return True, ""

    async def _check_daily_loss(self, portfolio: PortfolioState) -> tuple[bool, str]:
        if portfolio.daily_loss_pct >= self.config.max_daily_loss_pct:
            return False, f"Daily loss {portfolio.daily_loss_pct:.1%} >= limit {self.config.max_daily_loss_pct:.1%}"
        return True, ""

    async def _check_position_count(self, portfolio: PortfolioState) -> tuple[bool, str]:
        if portfolio.position_count >= self.config.max_positions:
            return False, f"Max positions reached: {portfolio.position_count}/{self.config.max_positions}"
        return True, ""

    async def _check_edge(self, prediction: EnsembleResult) -> tuple[bool, str]:
        if prediction.abs_edge < self.config.min_edge:
            return False, f"Edge {prediction.abs_edge:.3f} < minimum {self.config.min_edge}"
        return True, ""

    async def _check_confidence(self, prediction: EnsembleResult) -> tuple[bool, str]:
        if prediction.final_confidence < self.config.min_confidence:
            return False, f"Confidence {prediction.final_confidence:.3f} < minimum {self.config.min_confidence}"
        return True, ""

    async def _check_concentration(self, market: UnifiedMarket, portfolio: PortfolioState) -> tuple[bool, str]:
        if market.category:
            exposure = portfolio.category_exposure(market.category)
            if exposure >= self.config.max_sector_concentration:
                return False, f"Category '{market.category}' exposure {exposure:.1%} >= limit {self.config.max_sector_concentration:.1%}"
        return True, ""

    async def _check_cluster_limit(self, market: UnifiedMarket, portfolio: PortfolioState) -> tuple[bool, str]:
        """Cap positions per correlated cluster (e.g. same-league-same-night sports).

        Holding 10 NBA Finals winner contracts is really 1 bet on "the league",
        not 10 independent bets. This check enforces a per-cluster position cap
        that's tighter than the per-category exposure rule above.
        """
        new_cluster = classify_market(market).cluster
        max_per = getattr(self.config, "max_positions_per_cluster", 2)

        count_in_cluster = 0
        for pos in portfolio.positions:
            existing_cluster = cluster_key_from_title(pos.title, pos.category, pos.platform)
            if existing_cluster == new_cluster:
                count_in_cluster += 1

        if count_in_cluster >= max_per:
            return False, (
                f"Cluster '{new_cluster}' already has {count_in_cluster} "
                f"position(s), limit is {max_per}"
            )
        return True, ""

    def calculate_var_95(self, portfolio: PortfolioState) -> float:
        """
        Simplified Value at Risk at 95% confidence.
        Uses position sizes and a fixed volatility estimate.
        """
        if not portfolio.positions:
            return 0.0

        # Simple approach: sum of position-level VaR
        total_var = 0.0
        for pos in portfolio.positions:
            # Assume ~20% daily vol for prediction markets (they're binary)
            position_vol = pos.cost_basis * 0.20
            position_var = position_vol * 1.645  # 95% z-score
            total_var += position_var ** 2

        return math.sqrt(total_var)
