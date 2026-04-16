"""NewsAlpha main loop — LOG-ONLY MODE for initial validation.

Phase 1: observe and log divergence signals. No trading. This lets us verify:
    - Coinbase WS gives us live BTC prices
    - Polymarket short-duration BTC markets are discoverable
    - Our fair-value math produces reasonable numbers
    - Signals actually fire with expected frequency
    - End-to-end latency (spot tick → signal logged) is <1s

After 72h of log-only runs with good signal quality + quantity, we flip to
trading mode (Phase 2 — adds executor + tight exit logic).

The loop:
    - Coinbase WS streams in the background, updating self._spot
    - Every POLL_SECONDS, fetch all active Polymarket BTC markets
    - For each quote, run the divergence detector using current spot
    - Log qualifying signals to DB + structured logger
"""

from __future__ import annotations

import asyncio
import signal as os_signal
from datetime import datetime

import os

from src.newsalpha.coinbase_ws import CoinbaseTickerStream
from src.newsalpha.db import NewsAlphaDB
from src.newsalpha.executor import NewsAlphaExecutor, NewsAlphaExecutorConfig
from src.newsalpha.flash_detector import FlashMoveDetector
from src.newsalpha.news_classifier import NewsClassifier
from src.newsalpha.news_feed import NewsFeed
from src.newsalpha.polymarket_crypto import PolymarketCryptoFeed
from src.newsalpha.signal import SignalConfig, SignalGate, detect_divergence
from src.utils.logging import get_logger, setup_logging

logger = get_logger("newsalpha.orchestrator")


async def run_newsalpha(
    db_path: str = "newsalpha.db",
    poll_seconds: int = 3,
    log_level: str = "INFO",
    min_edge: float = 0.03,
    symbols: list[str] | None = None,
    observe_only: bool = False,
    bankroll: float = 1000.0,
) -> None:
    """Main loop for NewsAlpha.

    Args:
        db_path: path to the separate NewsAlpha SQLite DB.
        poll_seconds: how often to re-fetch Polymarket quotes. BTC spot
            streams in continuously, so this only controls market-list refresh.
        min_edge: minimum |fair - market| gap to log a signal.
        symbols: which spot instruments to stream (default: BTC-USD).
        observe_only: if True, log signals but do NOT trade. Always True for v1.
    """
    setup_logging(log_level)
    logger.info(
        "newsalpha_starting",
        poll_seconds=poll_seconds,
        min_edge=min_edge,
        observe_only=observe_only,
        db_path=db_path,
    )

    db = NewsAlphaDB(db_path)
    await db.initialize()

    # Initialize components BEFORE the ticker (which references flash.on_tick)
    poly_feed = PolymarketCryptoFeed()
    signal_cfg = SignalConfig(min_edge=min_edge)
    # Reduced-threshold config for flash-move state
    flash_signal_cfg = SignalConfig(min_edge=max(0.01, min_edge * 0.5))
    gate = SignalGate(cooldown_seconds=300.0, edge_jump_threshold=0.05)
    flash = FlashMoveDetector()
    executor_cfg = NewsAlphaExecutorConfig(bankroll=bankroll, paper_mode=True)
    executor = NewsAlphaExecutor(executor_cfg, db)

    # News feed + classifier (Haiku for speed)
    news_feed = NewsFeed()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    news_classifier = NewsClassifier(openrouter_key) if openrouter_key else None
    news_poll_interval = 60  # poll RSS every 60s (not every cycle — too aggressive)
    last_news_poll: float = 0.0
    news_boost_active = False
    news_boost_until: float = 0.0
    # News-boosted signal config: even lower threshold than flash
    news_signal_cfg = SignalConfig(min_edge=max(0.005, min_edge * 0.3))

    logger.info(
        "executor_initialized",
        bankroll=bankroll,
        observe_only=observe_only,
        news_enabled=news_classifier is not None,
    )

    # Start Coinbase WS — flash detector hooks into every tick
    ticker = CoinbaseTickerStream(
        symbols=symbols or ["BTC-USD"],
        on_tick=flash.on_tick,
    )
    await ticker.start()

    # Wait for the first BTC price so our first signals aren't garbage
    logger.info("waiting_for_first_tick")
    if not await ticker.wait_for_first_tick("BTC-USD", timeout=30.0):
        logger.error("first_tick_timeout — Coinbase WS connected but no ticker messages.")
        await ticker.stop()
        await db.close()
        return
    logger.info("first_tick_received", price=ticker.get_price("BTC-USD"))

    shutdown_event = asyncio.Event()

    def _handle_signal(sig, frame):
        logger.info("newsalpha_shutdown_signal", signal=sig)
        shutdown_event.set()

    os_signal.signal(os_signal.SIGINT, _handle_signal)
    os_signal.signal(os_signal.SIGTERM, _handle_signal)

    cycle = 0
    try:
        while not shutdown_event.is_set():
            cycle += 1
            cycle_start = datetime.utcnow()
            import time as _time
            now_mono = _time.monotonic()

            spot = ticker.get_price("BTC-USD")
            if spot is None:
                logger.warning("no_spot_available_skipping_cycle", cycle=cycle)
                await asyncio.sleep(poll_seconds)
                continue

            # --- News polling (every news_poll_interval seconds) ---
            if news_classifier and (now_mono - last_news_poll) >= news_poll_interval:
                last_news_poll = now_mono
                try:
                    headlines = await news_feed.poll()
                    if headlines:
                        classifications = await news_classifier.classify_batch(headlines[:5])
                        actionable = [c for c in classifications if c.is_actionable]
                        if actionable:
                            # Activate news boost for 120 seconds
                            news_boost_active = True
                            news_boost_until = now_mono + 120.0
                            best = max(actionable, key=lambda c: c.magnitude)
                            logger.info(
                                "news_boost_activated",
                                headline=best.headline[:60],
                                sentiment=best.sentiment,
                                magnitude=best.magnitude,
                                direction=best.direction,
                                actionable_count=len(actionable),
                            )
                except Exception as e:
                    logger.warning("news_poll_error", error=str(e))

            # Check if news boost has expired
            if news_boost_active and now_mono > news_boost_until:
                news_boost_active = False

            try:
                quotes = await poly_feed.fetch_active_btc_markets(current_spot=spot)
            except Exception as e:
                logger.error("fetch_markets_error", cycle=cycle, error=str(e))
                quotes = []

            # Detect if we're in a flash-move or news-boost state.
            # Priority: news_boost > flash > normal
            in_flash = flash.is_flash_active("BTC-USD")
            in_boost = news_boost_active or in_flash
            if news_boost_active:
                active_cfg = news_signal_cfg  # lowest threshold
            elif in_flash:
                active_cfg = flash_signal_cfg
            else:
                active_cfg = signal_cfg

            if in_boost and cycle % 10 == 0:  # don't spam log
                fe = flash.last_flash("BTC-USD")
                logger.info(
                    "boost_state_active",
                    flash=in_flash,
                    news=news_boost_active,
                    direction=fe.direction if fe else "news",
                    min_edge_lowered=active_cfg.min_edge,
                )

            signals = 0
            suppressed = 0
            for q in quotes:
                sig_obj = detect_divergence(q, spot, config=active_cfg)
                if sig_obj is None:
                    continue
                # In boost state (flash or news), bypass the gate — act now or miss it
                if not in_boost and not gate.should_emit(sig_obj):
                    suppressed += 1
                    continue
                signals += 1
                await db.log_signal(sig_obj)
                logger.info(
                    "divergence_signal",
                    market=sig_obj.title[:50],
                    side=sig_obj.side,
                    market_price=round(sig_obj.market_price, 3),
                    fair=round(sig_obj.fair_value, 3),
                    edge=round(sig_obj.edge, 3),
                    spot=round(sig_obj.spot_reference, 2),
                    ref=round(sig_obj.spot_at_window_start, 2),
                    sec_left=round(sig_obj.seconds_remaining, 0),
                    flash=in_flash,
                )
                # Execute if not in observe-only mode
                if not observe_only:
                    await executor.on_signal(sig_obj)

            # Check exits on existing positions (time-stop, stop-loss, trailing lock)
            if not observe_only and executor.open_position_count > 0:
                quotes_dict = {q.market_id: q for q in quotes}
                exits = await executor.check_exits(quotes_dict)
            else:
                exits = 0

            elapsed_ms = (datetime.utcnow() - cycle_start).total_seconds() * 1000
            summary = executor.get_summary() if not observe_only else {}
            logger.info(
                "cycle",
                cycle=cycle,
                btc_spot=round(spot, 2),
                markets_checked=len(quotes),
                signals_emitted=signals,
                suppressed_by_gate=suppressed,
                flash_active=in_flash,
                news_boost=news_boost_active,
                open_positions=summary.get("open_positions", 0),
                total_pnl=f"${summary.get('total_pnl', 0):+.2f}" if summary else "$0.00",
                exits=exits,
                cycle_ms=int(elapsed_ms),
            )

            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass

    finally:
        logger.info("newsalpha_shutting_down")
        await ticker.stop()
        await poly_feed.close()
        await news_feed.close()
        if news_classifier:
            await news_classifier.close()
        await db.close()
        logger.info("newsalpha_stopped")
