"""Separate DB for NewsAlpha. Default path: newsalpha.db in CWD.

Tables (all prefixed 'na_' to disambiguate if the same file ever merges):
    na_signals         → every divergence signal we log (pre-trade)
    na_positions       → currently open positions
    na_trades          → closed positions (realized PnL)
    na_ticks           → sampled BTC spot price history (optional, for audit)
    na_daily_stats     → per-day performance roll-up
"""

from __future__ import annotations

from datetime import datetime

import aiosqlite


SCHEMA: dict[str, str] = {
    "na_signals": """
        CREATE TABLE IF NOT EXISTS na_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            title TEXT NOT NULL,
            side TEXT NOT NULL,
            market_price REAL NOT NULL,
            fair_value REAL NOT NULL,
            edge REAL NOT NULL,
            seconds_remaining REAL NOT NULL,
            spot_reference REAL NOT NULL,
            spot_at_window_start REAL NOT NULL,
            acted_on INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "na_positions": """
        CREATE TABLE IF NOT EXISTS na_positions (
            position_id TEXT PRIMARY KEY,
            market_id TEXT NOT NULL,
            title TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            size REAL NOT NULL,
            cost_basis REAL NOT NULL,
            window_end TEXT NOT NULL,
            signal_edge REAL,
            is_paper INTEGER DEFAULT 1,
            opened_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "na_trades": """
        CREATE TABLE IF NOT EXISTS na_trades (
            trade_id TEXT PRIMARY KEY,
            market_id TEXT NOT NULL,
            title TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            size REAL NOT NULL,
            pnl REAL NOT NULL,
            pnl_pct REAL NOT NULL,
            hold_seconds REAL NOT NULL,
            outcome TEXT NOT NULL,
            exit_reason TEXT NOT NULL,
            signal_edge REAL,
            is_paper INTEGER DEFAULT 1,
            opened_at TEXT,
            closed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "na_ticks": """
        CREATE TABLE IF NOT EXISTS na_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "na_daily_stats": """
        CREATE TABLE IF NOT EXISTS na_daily_stats (
            date TEXT PRIMARY KEY,
            signals INTEGER DEFAULT 0,
            trades_opened INTEGER DEFAULT 0,
            trades_closed INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            pnl REAL DEFAULT 0,
            ai_cost REAL DEFAULT 0
        )
    """,
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_na_signals_time ON na_signals(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_na_signals_market ON na_signals(market_id)",
    "CREATE INDEX IF NOT EXISTS idx_na_trades_closed ON na_trades(closed_at)",
    "CREATE INDEX IF NOT EXISTS idx_na_ticks_time ON na_ticks(timestamp)",
]


class NewsAlphaDB:
    """Async DB for NewsAlpha. Intentionally mirrors the shape of the main
    DatabaseManager but never touches the same file by default."""

    def __init__(self, path: str = "newsalpha.db"):
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        for sql in SCHEMA.values():
            await self._db.execute(sql)
        for sql in INDEXES:
            await self._db.execute(sql)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("NewsAlphaDB not initialized")
        return self._db

    # --- Signals ---

    async def log_signal(self, signal) -> None:
        await self.db.execute(
            """INSERT INTO na_signals (market_id, title, side, market_price, fair_value,
               edge, seconds_remaining, spot_reference, spot_at_window_start, acted_on, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                signal.market_id, signal.title, signal.side,
                signal.market_price, signal.fair_value, signal.edge,
                signal.seconds_remaining, signal.spot_reference,
                signal.spot_at_window_start, signal.timestamp.isoformat(),
            ),
        )
        await self.db.commit()

    async def recent_signals(self, limit: int = 50) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM na_signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def count_signals_today(self) -> int:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM na_signals WHERE timestamp >= ?", (today,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    # --- Ticks (optional sampling; disabled by default) ---

    async def log_tick(self, symbol: str, price: float, source: str) -> None:
        await self.db.execute(
            "INSERT INTO na_ticks (symbol, price, source) VALUES (?, ?, ?)",
            (symbol, price, source),
        )
        # Commit is batched — caller decides when to flush via commit()
