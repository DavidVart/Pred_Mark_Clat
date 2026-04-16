"""Market-pair registry — defines which Polymarket ↔ Kalshi markets are 'the same event'.

Critical safety property: both markets must resolve IDENTICALLY on the same
underlying truth. A misaligned pair causes a guaranteed loss (both legs pay
$0 if their resolutions disagree).

We keep pairs in the database to make them editable without redeploy. A JSON
seed file (configs/market_pairs.json) populates the table on startup if it's
empty.

Currently pairs are curated by hand. Automatic matching (by date + keyword
similarity) is a future enhancement but too risky without human review —
the cost of a bad match (both legs losing) is far higher than the benefit
of one extra pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from src.arbitrage.models import MarketPair
from src.db.manager import DatabaseManager
from src.utils.logging import get_logger

logger = get_logger("arb.matcher")


# SQL for the market_pairs table — loaded alongside main schema.
MARKET_PAIRS_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_pairs (
    pair_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    polymarket_market_id TEXT NOT NULL,
    kalshi_ticker TEXT NOT NULL,
    category TEXT DEFAULT '',
    expected_resolution_date TEXT,
    mechanical_resolution INTEGER DEFAULT 1,
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    active INTEGER DEFAULT 1
)
"""


class MarketPairRegistry:
    """Async CRUD layer over the market_pairs table."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    async def initialize(self) -> None:
        await self.db.db.execute(MARKET_PAIRS_SCHEMA)
        await self.db.db.commit()

    async def seed_from_file(self, path: str | Path) -> int:
        """Load pairs from a JSON file. Upserts; returns rows affected."""
        p = Path(path)
        if not p.exists():
            logger.warning("seed_file_missing", path=str(p))
            return 0

        try:
            data = json.loads(p.read_text())
        except Exception as e:
            logger.error("seed_file_parse_error", path=str(p), error=str(e))
            return 0

        count = 0
        for entry in data.get("pairs", []):
            try:
                pair = MarketPair(**entry)
                await self.upsert(pair)
                count += 1
            except Exception as e:
                logger.error("seed_pair_invalid", entry=str(entry)[:80], error=str(e))

        logger.info("seeded_pairs", count=count, file=str(p))
        return count

    async def upsert(self, pair: MarketPair) -> None:
        await self.db.db.execute(
            """INSERT INTO market_pairs (
                pair_id, description, polymarket_market_id, kalshi_ticker,
                category, expected_resolution_date, mechanical_resolution, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pair_id) DO UPDATE SET
                description=excluded.description,
                polymarket_market_id=excluded.polymarket_market_id,
                kalshi_ticker=excluded.kalshi_ticker,
                category=excluded.category,
                expected_resolution_date=excluded.expected_resolution_date,
                mechanical_resolution=excluded.mechanical_resolution,
                notes=excluded.notes""",
            (
                pair.pair_id,
                pair.description,
                pair.polymarket_market_id,
                pair.kalshi_ticker,
                pair.category,
                pair.expected_resolution_date.isoformat() if pair.expected_resolution_date else None,
                1 if pair.mechanical_resolution else 0,
                pair.notes,
            ),
        )
        await self.db.db.commit()

    async def list_active(self) -> list[MarketPair]:
        cursor = await self.db.db.execute(
            "SELECT * FROM market_pairs WHERE active = 1"
        )
        rows = await cursor.fetchall()
        return [self._row_to_pair(r) for r in rows]

    async def get(self, pair_id: str) -> MarketPair | None:
        cursor = await self.db.db.execute(
            "SELECT * FROM market_pairs WHERE pair_id = ?", (pair_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_pair(row) if row else None

    async def deactivate(self, pair_id: str, reason: str = "") -> None:
        """Disable a pair without deleting it (e.g. after a resolution drift)."""
        await self.db.db.execute(
            "UPDATE market_pairs SET active = 0, notes = notes || ' [DEACTIVATED: ' || ? || ']' WHERE pair_id = ?",
            (reason, pair_id),
        )
        await self.db.db.commit()
        logger.warning("pair_deactivated", pair_id=pair_id, reason=reason)

    @staticmethod
    def _row_to_pair(row: aiosqlite.Row | dict | tuple) -> MarketPair:
        d = dict(row) if not isinstance(row, dict) else row
        from datetime import datetime as _dt
        exp = d.get("expected_resolution_date")
        return MarketPair(
            pair_id=d["pair_id"],
            description=d["description"],
            polymarket_market_id=d["polymarket_market_id"],
            kalshi_ticker=d["kalshi_ticker"],
            category=d.get("category") or "",
            expected_resolution_date=_dt.fromisoformat(exp) if exp else None,
            mechanical_resolution=bool(d.get("mechanical_resolution", 1)),
            notes=d.get("notes") or "",
        )
