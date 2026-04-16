"""Trade signal, execution, and position models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TradeSignal(BaseModel):
    """A recommended trade from the prediction pipeline."""

    market_id: str
    platform: str
    title: str
    side: Literal["yes", "no"]
    predicted_probability: float
    market_price: float
    edge: float
    confidence: float
    kelly_size: float  # Fraction of bankroll
    dollar_size: float = 0.0  # Actual dollar amount after sizing
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def abs_edge(self) -> float:
        return abs(self.edge)


class TradeExecution(BaseModel):
    """A completed trade (paper or live)."""

    execution_id: str
    signal: TradeSignal
    fill_price: float
    quantity: float
    total_cost: float
    is_paper: bool = True
    status: str = "filled"  # "filled", "partial", "rejected", "cancelled"
    platform_order_id: str = ""
    slippage: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Position(BaseModel):
    """An open position being tracked."""

    position_id: str
    market_id: str
    platform: str
    title: str
    side: Literal["yes", "no"]
    entry_price: float
    quantity: float
    cost_basis: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    is_paper: bool = True
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    category: str = ""

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "yes":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    @property
    def should_stop_loss(self) -> bool:
        return self.unrealized_pnl_pct <= -self.stop_loss if self.stop_loss > 0 else False

    @property
    def should_take_profit(self) -> bool:
        return self.unrealized_pnl_pct >= self.take_profit if self.take_profit > 0 else False


class ClosedTrade(BaseModel):
    """A resolved trade with final PnL."""

    trade_id: str
    market_id: str
    platform: str
    title: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    is_paper: bool
    category: str = ""
    outcome: str = ""  # "win", "loss"
    failure_class: str = ""  # "bad_prediction", "bad_timing", "bad_execution", "external_shock"
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: datetime = Field(default_factory=datetime.utcnow)
    hold_duration_hours: float = 0.0
