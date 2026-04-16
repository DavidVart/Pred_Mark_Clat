"""Portfolio state and risk metrics models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.models.trade import Position


class PortfolioState(BaseModel):
    """Current portfolio snapshot."""

    cash: float = 0.0
    positions: list[Position] = Field(default_factory=list)
    total_value: float = 0.0
    daily_pnl: float = 0.0
    daily_cost: float = 0.0
    peak_value: float = 0.0
    is_paper: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def total_exposure(self) -> float:
        return sum(p.cost_basis for p in self.positions)

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a positive fraction (0 = no drawdown)."""
        if self.peak_value <= 0:
            return 0.0
        return max(0.0, 1.0 - self.total_value / self.peak_value)

    @property
    def daily_loss_pct(self) -> float:
        if self.total_value <= 0:
            return 0.0
        return abs(min(0.0, self.daily_pnl)) / self.total_value

    def category_exposure(self, category: str) -> float:
        """Exposure to a specific category as fraction of total value."""
        if self.total_value <= 0:
            return 0.0
        cat_exposure = sum(
            p.cost_basis for p in self.positions if p.category == category
        )
        return cat_exposure / self.total_value


class RiskMetrics(BaseModel):
    """Computed risk metrics for the portfolio."""

    var_95: float = 0.0          # Value at Risk at 95% confidence
    max_drawdown: float = 0.0    # Peak-to-trough decline
    sharpe_ratio: float = 0.0    # Risk-adjusted return
    win_rate: float = 0.0        # Fraction of winning trades
    profit_factor: float = 0.0   # gross_profit / gross_loss
    brier_score: float = 0.0     # Prediction calibration
    total_trades: int = 0
    avg_hold_hours: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DailyStats(BaseModel):
    """Daily performance summary."""

    date: str  # YYYY-MM-DD
    pnl: float = 0.0
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    ai_cost: float = 0.0
    max_drawdown: float = 0.0
