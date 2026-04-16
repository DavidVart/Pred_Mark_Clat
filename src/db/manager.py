"""Async SQLite database manager."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import aiosqlite

from src.db.schema import INDEXES, TABLES


class DatabaseManager:
    """Async database manager wrapping aiosqlite."""

    def __init__(self, db_path: str = "trading.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create all tables and indexes."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        for table_sql in TABLES.values():
            await self._db.execute(table_sql)
        for index_sql in INDEXES:
            await self._db.execute(index_sql)

        # Ensure kill_switch has exactly one row
        await self._db.execute(
            "INSERT OR IGNORE INTO kill_switch (id, active) VALUES (1, 0)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._db

    # --- Markets ---

    async def upsert_market(self, market: dict) -> None:
        await self.db.execute(
            """INSERT INTO markets (market_id, platform, title, description, category,
               yes_price, no_price, volume, liquidity, expiration, status, outcomes,
               url, ticker, clob_token_ids, condition_id, last_updated)
               VALUES (:market_id, :platform, :title, :description, :category,
               :yes_price, :no_price, :volume, :liquidity, :expiration, :status,
               :outcomes, :url, :ticker, :clob_token_ids, :condition_id, :last_updated)
               ON CONFLICT(market_id) DO UPDATE SET
               yes_price=:yes_price, no_price=:no_price, volume=:volume,
               liquidity=:liquidity, status=:status, last_updated=:last_updated""",
            {
                **market,
                "last_updated": datetime.utcnow().isoformat(),
                "outcomes": json.dumps(market.get("outcomes", ["Yes", "No"])),
                "clob_token_ids": json.dumps(market.get("clob_token_ids")),
            },
        )
        await self.db.commit()

    async def get_market(self, market_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM markets WHERE market_id = ?", (market_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # --- Positions ---

    async def insert_position(self, position: dict) -> None:
        await self.db.execute(
            """INSERT INTO positions (position_id, market_id, platform, title, side,
               entry_price, quantity, cost_basis, current_price, stop_loss, take_profit,
               is_paper, category, opened_at)
               VALUES (:position_id, :market_id, :platform, :title, :side,
               :entry_price, :quantity, :cost_basis, :current_price, :stop_loss,
               :take_profit, :is_paper, :category, :opened_at)""",
            position,
        )
        await self.db.commit()

    async def get_open_positions(self) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM positions")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_position_price(self, position_id: str, price: float) -> None:
        await self.db.execute(
            "UPDATE positions SET current_price = ? WHERE position_id = ?",
            (price, position_id),
        )
        await self.db.commit()

    async def close_position(self, position_id: str, exit_price: float, pnl: float, pnl_pct: float, outcome: str = "", failure_class: str = "") -> None:
        """Move position to trade_log and delete from positions."""
        cursor = await self.db.execute(
            "SELECT * FROM positions WHERE position_id = ?", (position_id,)
        )
        pos = await cursor.fetchone()
        if not pos:
            return

        pos = dict(pos)
        now = datetime.utcnow()
        opened = datetime.fromisoformat(pos["opened_at"]) if pos["opened_at"] else now
        hold_hours = (now - opened).total_seconds() / 3600

        await self.db.execute(
            """INSERT INTO trade_log (trade_id, market_id, platform, title, side,
               entry_price, exit_price, quantity, pnl, pnl_pct, is_paper, category,
               outcome, failure_class, opened_at, closed_at, hold_duration_hours)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pos["position_id"], pos["market_id"], pos["platform"], pos["title"],
                pos["side"], pos["entry_price"], exit_price, pos["quantity"],
                pnl, pnl_pct, pos["is_paper"], pos["category"],
                outcome, failure_class, pos["opened_at"], now.isoformat(), hold_hours,
            ),
        )
        await self.db.execute(
            "DELETE FROM positions WHERE position_id = ?", (position_id,)
        )
        await self.db.commit()

    # --- Predictions ---

    async def log_prediction(self, prediction: dict) -> None:
        await self.db.execute(
            """INSERT INTO predictions (market_id, weighted_probability, final_confidence,
               disagreement_score, edge, models_succeeded, models_failed, total_cost_usd,
               individual_json)
               VALUES (:market_id, :weighted_probability, :final_confidence,
               :disagreement_score, :edge, :models_succeeded, :models_failed,
               :total_cost_usd, :individual_json)""",
            prediction,
        )
        await self.db.commit()

    # --- LLM Cost Tracking ---

    async def log_llm_query(self, query: dict) -> None:
        await self.db.execute(
            """INSERT INTO llm_queries (model_name, role, market_id, prompt_tokens,
               completion_tokens, total_tokens, cost_usd, duration_ms, success, error)
               VALUES (:model_name, :role, :market_id, :prompt_tokens,
               :completion_tokens, :total_tokens, :cost_usd, :duration_ms,
               :success, :error)""",
            query,
        )
        await self.db.commit()

    async def get_daily_ai_cost(self) -> float:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_queries WHERE timestamp >= ?",
            (today,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0.0

    # --- Kill Switch ---

    async def get_kill_switch(self) -> dict:
        cursor = await self.db.execute("SELECT * FROM kill_switch WHERE id = 1")
        row = await cursor.fetchone()
        return dict(row) if row else {"active": 0, "reason": ""}

    async def set_kill_switch(self, active: bool, reason: str = "", activated_by: str = "system") -> None:
        await self.db.execute(
            """UPDATE kill_switch SET active = ?, reason = ?, activated_at = ?,
               activated_by = ? WHERE id = 1""",
            (int(active), reason, datetime.utcnow().isoformat() if active else None, activated_by),
        )
        await self.db.commit()

    async def is_kill_switch_active(self) -> bool:
        ks = await self.get_kill_switch()
        return bool(ks.get("active", 0))

    # --- Analysis Cooldown ---

    async def record_analysis(self, market_id: str) -> None:
        now = datetime.utcnow().isoformat()
        await self.db.execute(
            "INSERT INTO analysis_cooldown (market_id, analyzed_at) VALUES (?, ?)",
            (market_id, now),
        )
        await self.db.commit()

    async def is_on_cooldown(self, market_id: str, cooldown_hours: float = 4.0) -> bool:
        cutoff = (datetime.utcnow() - timedelta(hours=cooldown_hours)).isoformat()
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM analysis_cooldown WHERE market_id = ? AND analyzed_at > ?",
            (market_id, cutoff),
        )
        row = await cursor.fetchone()
        return (row[0] if row else 0) > 0

    async def get_daily_analysis_count(self, market_id: str) -> int:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM analysis_cooldown WHERE market_id = ? AND analyzed_at >= ?",
            (market_id, today),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    # --- Trade Log Queries ---

    async def get_trade_history(self, limit: int = 50) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM trade_log ORDER BY closed_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_daily_pnl(self) -> float:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trade_log WHERE closed_at >= ?",
            (today,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0.0

    async def get_win_rate(self) -> float:
        cursor = await self.db.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins FROM trade_log"
        )
        row = await cursor.fetchone()
        if not row or row[0] == 0:
            return 0.0
        return row[1] / row[0]

    # --- Daily Stats ---

    async def upsert_daily_stats(self, stats: dict) -> None:
        await self.db.execute(
            """INSERT INTO daily_stats (date, pnl, trades_opened, trades_closed, wins,
               losses, ai_cost, max_drawdown)
               VALUES (:date, :pnl, :trades_opened, :trades_closed, :wins,
               :losses, :ai_cost, :max_drawdown)
               ON CONFLICT(date) DO UPDATE SET
               pnl=:pnl, trades_opened=:trades_opened, trades_closed=:trades_closed,
               wins=:wins, losses=:losses, ai_cost=:ai_cost, max_drawdown=:max_drawdown""",
            stats,
        )
        await self.db.commit()
