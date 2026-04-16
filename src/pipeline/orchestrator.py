"""Main orchestrator — wires all pipeline stages and runs the bot loop."""

from __future__ import annotations

import asyncio
import signal
import sys

from config.models import get_ensemble as get_model_specs
from config.settings import Settings
from src.agents.ensemble import EnsembleRunner, build_agents
from src.clients.kalshi_client import KalshiClient
from src.clients.openrouter_client import OpenRouterClient
from src.clients.polymarket_client import PolymarketClient
from src.db.manager import DatabaseManager
from src.pipeline.calibration import get_current_scaler
from src.pipeline.compounder import OutcomeTracker
from src.pipeline.executor import TradeExecutor
from src.pipeline.predictor import MarketPredictor
from src.pipeline.researcher import MarketResearcher
from src.pipeline.risk_manager import RiskManager
from src.pipeline.scanner import MarketScanner
from src.utils import kill_switch
from src.utils.cost_tracker import CostTracker
from src.utils.logging import get_logger, setup_logging

logger = get_logger("orchestrator")


async def run_bot(
    settings: Settings,
    paper_mode: bool = True,
    platforms: str = "both",
    interval: int = 5,
) -> None:
    """Main bot entry point — runs the Scan→Research→Predict→Risk→Execute→Compound loop."""
    setup_logging(settings.log_level)
    mode = "PAPER" if paper_mode else "LIVE"
    logger.info("bot_initializing", mode=mode, platforms=platforms, interval=interval)

    # --- Initialize components ---
    db = DatabaseManager(settings.db_path)
    await db.initialize()

    # Exchange clients
    clients: dict[str, PolymarketClient | KalshiClient] = {}
    if platforms in ("both", "polymarket"):
        clients["polymarket"] = PolymarketClient(
            wallet_private_key=settings.polymarket.wallet_private_key,
            live_mode=not paper_mode,
        )
        logger.info("client_ready", platform="polymarket")

    if platforms in ("both", "kalshi"):
        clients["kalshi"] = KalshiClient(
            api_key=settings.kalshi.api_key,
            private_key_path=settings.kalshi.private_key_path,
            use_demo=settings.kalshi.use_demo,
            live_mode=not paper_mode,
        )
        logger.info("client_ready", platform="kalshi")

    if not clients:
        logger.error("no_clients_configured")
        return

    # OpenRouter LLM client
    openrouter = OpenRouterClient(settings.openrouter_api_key)

    # Build ensemble
    model_specs = get_model_specs()
    agents = build_agents(openrouter, model_specs)
    ensemble = EnsembleRunner(
        agents=agents,
        min_models=settings.ensemble.min_models_for_consensus,
        disagreement_threshold=settings.ensemble.disagreement_threshold,
        disagreement_penalty=settings.ensemble.disagreement_penalty,
    )

    # Pipeline stages
    cost_tracker = CostTracker(db, settings.cost.daily_ai_cost_limit)
    scanner = MarketScanner(list(clients.values()), settings.scanner, db)
    researcher = MarketResearcher()

    # Fit Platt scaler from accumulated trade history. Returns identity (no-op)
    # if we don't have enough resolved trades yet (<30 samples).
    calibrator = await get_current_scaler(db)

    predictor = MarketPredictor(
        ensemble=ensemble,
        db=db,
        cost_tracker=cost_tracker,
        min_edge=settings.risk.min_edge,
        min_net_edge=settings.risk.min_net_edge,
        min_confidence=settings.risk.min_confidence,
        max_analyses_per_day=settings.cost.max_analyses_per_market_per_day,
        prefer_maker_orders=settings.risk.prefer_maker_orders,
        calibrator=calibrator,
    )
    risk_mgr = RiskManager(settings.risk, db)
    executor = TradeExecutor(
        clients=clients,
        db=db,
        paper_mode=paper_mode,
        stop_loss_pct=settings.risk.stop_loss_pct,
        take_profit_pct=settings.risk.take_profit_pct,
        prefer_maker_orders=settings.risk.prefer_maker_orders,
    )
    compounder = OutcomeTracker(db)

    # --- Graceful shutdown ---
    shutdown_event = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info("shutdown_signal", signal=sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # --- Main loop ---
    cycle = 0
    logger.info("bot_started", mode=mode)

    try:
        while not shutdown_event.is_set():
            cycle += 1
            logger.info("cycle_start", cycle=cycle)

            try:
                await _run_cycle(
                    scanner, researcher, predictor, risk_mgr, executor, compounder, db
                )
            except Exception as e:
                logger.error("cycle_error", cycle=cycle, error=str(e))

            # Update daily stats
            try:
                await compounder.update_daily_stats()
            except Exception as e:
                logger.error("stats_error", error=str(e))

            # Wait for next cycle or shutdown
            logger.info("cycle_complete", cycle=cycle, next_in_minutes=interval)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=interval * 60)
            except asyncio.TimeoutError:
                pass  # Normal — timeout means it's time for next cycle

    finally:
        # Cleanup
        logger.info("bot_shutting_down")
        for client in clients.values():
            await client.close()
        await researcher.close()
        await openrouter.close()
        await db.close()
        logger.info("bot_stopped")


async def _run_cycle(
    scanner: MarketScanner,
    researcher: MarketResearcher,
    predictor: MarketPredictor,
    risk_mgr: RiskManager,
    executor: TradeExecutor,
    compounder: OutcomeTracker,
    db: DatabaseManager,
) -> None:
    """Execute one full pipeline cycle."""

    # Check kill switch
    halted, reason = await kill_switch.is_halted(db)
    if halted:
        logger.warning("cycle_halted", reason=reason)
        return

    # Stage 1: Scan
    logger.info("stage_scan")
    candidates = await scanner.scan()
    if not candidates:
        logger.info("no_candidates")
        return

    # Stage 2: Research
    logger.info("stage_research", candidates=len(candidates))
    enriched = await researcher.research_batch(candidates[:10])

    # Stage 3: Predict
    logger.info("stage_predict")
    signals = []
    for market, context in enriched:
        prediction = await predictor.predict(market, context)
        if prediction:
            signals.append((market, prediction))

    if not signals:
        logger.info("no_signals")
        # Still check exits even with no new signals
        await executor.check_exits()
        return

    # Stage 4: Risk
    logger.info("stage_risk", signals=len(signals))
    portfolio = await executor.get_portfolio_state()
    approved = []

    for market, prediction in signals:
        trade_signal, reason = await risk_mgr.approve_trade(market, prediction, portfolio)
        if trade_signal:
            approved.append(trade_signal)

    # Stage 5: Execute
    if approved:
        logger.info("stage_execute", approved=len(approved))
        for trade_signal in approved:
            await executor.execute(trade_signal)

    # Stage 5b: Check exits on existing positions
    exits = await executor.check_exits()

    logger.info(
        "cycle_summary",
        scanned=len(candidates),
        researched=len(enriched),
        predicted=len(signals),
        approved=len(approved),
        exits=exits,
    )
