"""Main loop for the arbitrage bot.

Much simpler than the LLM bot orchestrator:
    1. Initialize clients + DB + registry
    2. Every N seconds:
       a. Scan all active pairs for profitable spreads
       b. For each profitable opportunity: execute immediately
       c. No cooldown, no research, no LLM — speed matters, spreads collapse fast
    3. Graceful shutdown on SIGINT
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

from config.settings import Settings
from src.arbitrage.executor import ArbExecutor
from src.arbitrage.matcher import MarketPairRegistry
from src.arbitrage.scanner import ArbScanner
from src.clients.kalshi_client import KalshiClient
from src.clients.polymarket_client import PolymarketClient
from src.db.manager import DatabaseManager
from src.utils import kill_switch
from src.utils.logging import get_logger, setup_logging

logger = get_logger("arb.orchestrator")

DEFAULT_SEED_FILE = "configs/market_pairs.json"


async def run_arb(
    settings: Settings,
    paper_mode: bool = True,
    interval_seconds: int = 30,
    min_net_spread: float = 0.005,
    max_notional: float = 100.0,
    seed_file: str | None = None,
) -> None:
    """Main arbitrage loop."""
    setup_logging(settings.log_level)
    mode = "PAPER" if paper_mode else "LIVE"
    logger.info(
        "arb_bot_starting",
        mode=mode,
        interval_seconds=interval_seconds,
        min_net_spread=min_net_spread,
        max_notional=max_notional,
    )

    db = DatabaseManager(settings.db_path)
    await db.initialize()

    poly = PolymarketClient(
        wallet_private_key=settings.polymarket.wallet_private_key,
        live_mode=not paper_mode,
    )
    kalshi = KalshiClient(
        api_key=settings.kalshi.api_key,
        private_key_path=settings.kalshi.private_key_path,
        use_demo=settings.kalshi.use_demo,
        live_mode=not paper_mode,
    )

    registry = MarketPairRegistry(db)
    await registry.initialize()

    # Seed from JSON if present and the registry is empty
    existing = await registry.list_active()
    if not existing:
        seed_path = seed_file or DEFAULT_SEED_FILE
        if Path(seed_path).exists():
            await registry.seed_from_file(seed_path)
        else:
            logger.warning("no_pairs_and_no_seed", seed_file=seed_path)

    pairs = await registry.list_active()
    if not pairs:
        logger.error("no_active_pairs_cannot_run")
        await poly.close()
        await kalshi.close()
        await db.close()
        return

    logger.info("arb_pairs_loaded", count=len(pairs))
    for p in pairs:
        logger.info(
            "arb_pair",
            pair_id=p.pair_id,
            description=p.description[:60],
            category=p.category,
        )

    scanner = ArbScanner(
        poly=poly,
        kalshi=kalshi,
        registry=registry,
        min_net_spread=min_net_spread,
    )
    executor = ArbExecutor(
        poly=poly,
        kalshi=kalshi,
        db=db,
        paper_mode=paper_mode,
        max_notional_per_trade=max_notional,
    )
    await executor.initialize()

    shutdown_event = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info("arb_shutdown_signal", signal=sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cycle = 0
    try:
        while not shutdown_event.is_set():
            cycle += 1

            # Respect the shared kill switch — same one as the LLM bot
            halted, reason = await kill_switch.is_halted(db)
            if halted:
                logger.warning("arb_halted_by_kill_switch", reason=reason)
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
                except asyncio.TimeoutError:
                    pass
                continue

            try:
                opportunities = await scanner.scan()
                for opp in opportunities:
                    # Final check: profitability still above threshold after scan
                    if opp.net_spread < min_net_spread:
                        logger.info(
                            "opportunity_below_threshold",
                            pair=opp.pair_id,
                            net=f"{opp.net_spread:+.3f}",
                            threshold=min_net_spread,
                        )
                        continue
                    await executor.execute(opp, notional=max_notional)
            except Exception as e:
                logger.error("arb_cycle_error", cycle=cycle, error=str(e))

            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass

    finally:
        logger.info("arb_bot_shutting_down")
        await poly.close()
        await kalshi.close()
        await db.close()
        logger.info("arb_bot_stopped")
