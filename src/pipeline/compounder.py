"""Stage 6: Compounder — track outcomes, classify failures, update learnings."""

from __future__ import annotations

from datetime import datetime

from src.db.manager import DatabaseManager
from src.utils.logging import get_logger

logger = get_logger("compounder")


class OutcomeTracker:
    """Tracks settled markets, classifies outcomes, and updates performance metrics."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    async def update_daily_stats(self) -> dict:
        """Compute and store daily performance summary."""
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Get today's closed trades
        trades = await self.db.get_trade_history(limit=1000)
        today_trades = [t for t in trades if t.get("closed_at", "").startswith(today)]

        wins = sum(1 for t in today_trades if t["pnl"] > 0)
        losses = sum(1 for t in today_trades if t["pnl"] <= 0)
        pnl = sum(t["pnl"] for t in today_trades)
        ai_cost = await self.db.get_daily_ai_cost()

        # Count new positions opened today
        positions = await self.db.get_open_positions()
        opened_today = sum(
            1 for p in positions
            if p.get("opened_at", "").startswith(today)
        )

        stats = {
            "date": today,
            "pnl": pnl,
            "trades_opened": opened_today,
            "trades_closed": len(today_trades),
            "wins": wins,
            "losses": losses,
            "ai_cost": ai_cost,
            "max_drawdown": 0.0,  # Updated by portfolio state
        }

        await self.db.upsert_daily_stats(stats)

        logger.info(
            "daily_stats",
            date=today,
            pnl=f"${pnl:+.2f}",
            trades=len(today_trades),
            wins=wins,
            losses=losses,
            ai_cost=f"${ai_cost:.2f}",
        )

        return stats

    async def compute_performance_metrics(self) -> dict:
        """Compute overall performance metrics."""
        trades = await self.db.get_trade_history(limit=10000)

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_pnl": 0.0,
                "avg_hold_hours": 0.0,
                "sharpe_ratio": 0.0,
            }

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))

        win_rate = len(wins) / len(trades) if trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_pnl = sum(t["pnl"] for t in trades) / len(trades)
        avg_hold = sum(t.get("hold_duration_hours", 0) for t in trades) / len(trades)

        # Simplified Sharpe: mean(returns) / std(returns)
        returns = [t["pnl_pct"] for t in trades if t.get("pnl_pct") is not None]
        if len(returns) > 1:
            mean_r = sum(returns) / len(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = variance ** 0.5
            sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0  # Annualized
        else:
            sharpe = 0.0

        metrics = {
            "total_trades": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_pnl": avg_pnl,
            "avg_hold_hours": avg_hold,
            "sharpe_ratio": sharpe,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }

        logger.info(
            "performance_metrics",
            trades=len(trades),
            win_rate=f"{win_rate:.1%}",
            profit_factor=f"{profit_factor:.2f}",
            sharpe=f"{sharpe:.2f}",
        )

        return metrics
