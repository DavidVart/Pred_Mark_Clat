"""Multi-trigger kill switch mechanism."""

from __future__ import annotations

from pathlib import Path

from src.db.manager import DatabaseManager
from src.utils.logging import get_logger

logger = get_logger("kill_switch")

KILL_FILE = Path("KILL")


async def is_halted(db: DatabaseManager) -> tuple[bool, str]:
    """Check all kill switch triggers. Returns (halted, reason)."""
    # 1. File sentinel
    if KILL_FILE.exists():
        reason = KILL_FILE.read_text().strip() or "KILL file detected"
        logger.warning("kill_switch_file", reason=reason)
        return True, reason

    # 2. Database flag
    if await db.is_kill_switch_active():
        ks = await db.get_kill_switch()
        reason = ks.get("reason", "Kill switch active")
        logger.warning("kill_switch_db", reason=reason)
        return True, reason

    return False, ""


async def activate(db: DatabaseManager, reason: str, activated_by: str = "system") -> None:
    """Activate the kill switch."""
    await db.set_kill_switch(True, reason, activated_by)
    logger.warning("kill_switch_activated", reason=reason, by=activated_by)


async def deactivate(db: DatabaseManager) -> None:
    """Deactivate the kill switch."""
    await db.set_kill_switch(False)
    # Also remove file sentinel if present
    if KILL_FILE.exists():
        KILL_FILE.unlink()
    logger.info("kill_switch_deactivated")
