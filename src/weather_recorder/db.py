"""Weather-edge recorder DB schema."""

from __future__ import annotations

import aiosqlite

SCHEMA: dict[str, str] = {
    "wr_snapshots": """
        CREATE TABLE IF NOT EXISTS wr_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_time TEXT DEFAULT CURRENT_TIMESTAMP,
            market_ticker TEXT NOT NULL,
            event_ticker TEXT NOT NULL,
            event_date TEXT NOT NULL,       -- YYYY-MM-DD, the day the market resolves on
            city TEXT NOT NULL,              -- NYC, LAX, CHI, MIA, AUS
            direction TEXT NOT NULL,         -- above | below | between
            threshold REAL NOT NULL,
            -- Market pricing
            yes_best_bid REAL,
            yes_best_ask REAL,
            yes_mid REAL,
            yes_book_depth REAL,             -- total contracts bid+asked near top of book
            -- Our model's prediction
            our_yes_prob REAL NOT NULL,
            noaa_high_forecast REAL,
            noaa_sigma REAL DEFAULT 3.0,
            -- Outcome (filled in after resolution)
            resolved INTEGER DEFAULT 0,
            actual_high REAL,
            outcome_yes INTEGER              -- 1 if YES resolved, 0 if NO, NULL if not yet
        )
    """,
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_wr_snap_time ON wr_snapshots(snapshot_time)",
    "CREATE INDEX IF NOT EXISTS idx_wr_snap_ticker ON wr_snapshots(market_ticker)",
    "CREATE INDEX IF NOT EXISTS idx_wr_snap_event ON wr_snapshots(event_ticker, event_date)",
    "CREATE INDEX IF NOT EXISTS idx_wr_snap_resolved ON wr_snapshots(resolved)",
]


class WeatherRecorderDB:
    def __init__(self, path: str = "weather_recorder.db"):
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
            raise RuntimeError("WeatherRecorderDB not initialized")
        return self._db

    async def log_snapshot(self, snapshot: dict) -> None:
        await self.db.execute(
            """INSERT INTO wr_snapshots (
                market_ticker, event_ticker, event_date, city, direction, threshold,
                yes_best_bid, yes_best_ask, yes_mid, yes_book_depth,
                our_yes_prob, noaa_high_forecast, noaa_sigma
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot["market_ticker"], snapshot["event_ticker"],
                snapshot["event_date"], snapshot["city"], snapshot["direction"],
                snapshot["threshold"],
                snapshot.get("yes_best_bid"), snapshot.get("yes_best_ask"),
                snapshot.get("yes_mid"), snapshot.get("yes_book_depth"),
                snapshot["our_yes_prob"], snapshot.get("noaa_high_forecast"),
                snapshot.get("noaa_sigma", 3.0),
            ),
        )
        await self.db.commit()

    async def mark_resolved(self, market_ticker: str, actual_high: float, outcome_yes: int) -> int:
        """Mark all snapshots for a market as resolved."""
        cursor = await self.db.execute(
            """UPDATE wr_snapshots SET resolved = 1, actual_high = ?, outcome_yes = ?
               WHERE market_ticker = ? AND resolved = 0""",
            (actual_high, outcome_yes, market_ticker),
        )
        await self.db.commit()
        return cursor.rowcount
