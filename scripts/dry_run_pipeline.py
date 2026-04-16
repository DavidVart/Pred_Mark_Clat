"""Zero-cost validation of the post-ensemble pipeline.

Feeds mock ensemble results through RiskManager -> TradeExecutor -> DB,
then simulates price movement to verify check_exits() correctly triggers
take-profit and stop-loss closures. Uses a MockExchangeClient instead of
real Polymarket/Kalshi to avoid any network calls or real fills.

Run: PYTHONUNBUFFERED=1 python -u scripts/dry_run_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from config.settings import RiskConfig
from src.db.manager import DatabaseManager
from src.models.market import OrderBook, UnifiedMarket
from src.models.portfolio import PortfolioState
from src.models.prediction import AgentPrediction, EnsembleResult
from src.models.trade import Position
from src.pipeline.executor import TradeExecutor
from src.pipeline.risk_manager import RiskManager
from src.utils.logging import setup_logging


class MockExchangeClient:
    """In-memory stand-in for PolymarketClient / KalshiClient."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self._markets: dict[str, UnifiedMarket] = {}
        self._order_counter = 0

    @property
    def platform_name(self) -> str:
        return self._name

    def set_market(self, market: UnifiedMarket) -> None:
        self._markets[market.market_id] = market

    def move_price(self, market_id: str, new_yes_price: float) -> None:
        """Simulate the market moving — used to trigger take-profit / stop-loss."""
        m = self._markets[market_id]
        self._markets[market_id] = m.model_copy(
            update={"yes_price": new_yes_price, "no_price": 1.0 - new_yes_price}
        )

    def close_market(self, market_id: str, final_yes_price: float) -> None:
        """Simulate market settlement."""
        m = self._markets[market_id]
        self._markets[market_id] = m.model_copy(
            update={
                "yes_price": final_yes_price,
                "no_price": 1.0 - final_yes_price,
                "status": "closed",
            }
        )

    async def get_active_markets(self, limit: int = 100) -> list[UnifiedMarket]:
        return list(self._markets.values())

    async def get_market(self, market_id: str) -> UnifiedMarket | None:
        return self._markets.get(market_id)

    async def get_orderbook(self, market_id: str) -> OrderBook:
        return OrderBook(market_id=market_id)

    async def place_order(self, market_id: str, side: str, size: float, price: float) -> str:
        self._order_counter += 1
        return f"{self._name}-order-{self._order_counter}"

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def get_balance(self) -> float:
        return 1000.0

    async def get_positions(self) -> list[Position]:
        return []

    async def close(self) -> None:
        pass


def make_market(market_id: str, title: str, yes_price: float) -> UnifiedMarket:
    return UnifiedMarket(
        platform="polymarket",
        market_id=market_id,
        title=title,
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        volume=100_000,
        liquidity=10_000.0,
        expiration=datetime.utcnow() + timedelta(days=30),
        status="active",
    )


async def register_market(db: DatabaseManager, client: "MockExchangeClient", market: UnifiedMarket) -> None:
    """Put the market in both the mock client and the DB (positions has FK → markets)."""
    client.set_market(market)
    await db.upsert_market(market.model_dump(mode="json"))


def make_prediction(market_id: str, probability: float, confidence: float, market_price: float) -> EnsembleResult:
    """Build a fully-populated EnsembleResult from a desired probability."""
    preds = [
        AgentPrediction(
            model_name=f"mock-model-{i}",
            role=role,
            probability=probability + (i * 0.01 - 0.02),
            confidence=confidence,
            reasoning="mock",
            cost_usd=0.0,
            tokens_used=0,
        )
        for i, role in enumerate(
            ["forecaster", "news_analyst", "risk_manager", "bull_researcher", "bear_researcher"]
        )
    ]
    return EnsembleResult(
        market_id=market_id,
        weighted_probability=probability,
        final_confidence=confidence,
        individual_predictions=preds,
        disagreement_score=0.02,
        edge=probability - market_price,
        total_cost_usd=0.0,
        models_succeeded=5,
        models_failed=0,
    )


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


async def main() -> int:
    setup_logging("INFO")

    # Use a throwaway sqlite file so we don't pollute trading.db
    tmpdir = tempfile.mkdtemp(prefix="dryrun-")
    db_path = os.path.join(tmpdir, "dryrun.db")
    print(f"Using throwaway DB at: {db_path}")

    db = DatabaseManager(db_path)
    await db.initialize()

    client = MockExchangeClient("polymarket")
    executor = TradeExecutor(
        clients={"polymarket": client},
        db=db,
        paper_mode=True,
        stop_loss_pct=0.10,
        take_profit_pct=0.25,
    )
    risk_mgr = RiskManager(RiskConfig(), db)

    failures = 0

    # ---------------------------------------------------------------
    print_section("Step 1 — Trade with good edge should be APPROVED + EXECUTED")
    # ---------------------------------------------------------------
    mkt_a = make_market("MKT-A", "Mock Market A (strong YES edge)", yes_price=0.40)
    await register_market(db, client, mkt_a)

    # Models say 62% vs market 40% → edge +22%
    pred_a = make_prediction("MKT-A", probability=0.62, confidence=0.75, market_price=0.40)
    portfolio = await executor.get_portfolio_state(initial_capital=1000.0)

    signal_a, reason_a = await risk_mgr.approve_trade(mkt_a, pred_a, portfolio)
    print(f"  risk.approve_trade -> signal={'<signal>' if signal_a else None}, reason='{reason_a}'")
    if signal_a is None:
        print(f"  FAIL: expected approval, got rejection: {reason_a}")
        failures += 1
    else:
        print(f"  side={signal_a.side} edge={signal_a.edge:+.3f} "
              f"kelly={signal_a.kelly_size:.4f} dollar_size=${signal_a.dollar_size:.2f}")

        execution = await executor.execute(signal_a)
        if execution is None:
            print("  FAIL: executor.execute returned None")
            failures += 1
        else:
            print(f"  executor.execute -> order_id={execution.execution_id}")
            print(f"  fill_price={execution.fill_price:.3f} qty={execution.quantity:.2f}")

    # ---------------------------------------------------------------
    print_section("Step 2 — Confirm position is in the DB")
    # ---------------------------------------------------------------
    positions = await db.get_open_positions()
    print(f"  db.get_open_positions() -> {len(positions)} position(s)")
    for p in positions:
        print(f"    {p['position_id']}: {p['title'][:40]} | {p['side']} | "
              f"entry=${p['entry_price']:.3f} qty={p['quantity']:.2f} "
              f"cost=${p['cost_basis']:.2f} paper={p['is_paper']}")
    if len(positions) != 1:
        print(f"  FAIL: expected 1 open position, got {len(positions)}")
        failures += 1

    # ---------------------------------------------------------------
    print_section("Step 3 — Trade with edge below threshold should be REJECTED")
    # ---------------------------------------------------------------
    mkt_b = make_market("MKT-B", "Mock Market B (tiny edge)", yes_price=0.50)
    await register_market(db, client, mkt_b)

    pred_b = make_prediction("MKT-B", probability=0.52, confidence=0.80, market_price=0.50)
    portfolio = await executor.get_portfolio_state(initial_capital=1000.0)
    signal_b, reason_b = await risk_mgr.approve_trade(mkt_b, pred_b, portfolio)
    print(f"  signal={signal_b} reason='{reason_b}'")
    if signal_b is not None:
        print("  FAIL: expected rejection on tiny edge, got approval")
        failures += 1
    elif "Edge" not in reason_b and "edge" not in reason_b:
        print(f"  WARN: rejected but reason doesn't mention edge: {reason_b}")

    # ---------------------------------------------------------------
    print_section("Step 4 — Trade with low confidence should be REJECTED")
    # ---------------------------------------------------------------
    mkt_c = make_market("MKT-C", "Mock Market C (big edge, low conf)", yes_price=0.40)
    await register_market(db, client, mkt_c)

    pred_c = make_prediction("MKT-C", probability=0.65, confidence=0.30, market_price=0.40)
    signal_c, reason_c = await risk_mgr.approve_trade(
        mkt_c, pred_c, await executor.get_portfolio_state(initial_capital=1000.0)
    )
    print(f"  signal={signal_c} reason='{reason_c}'")
    if signal_c is not None:
        print("  FAIL: expected rejection on low confidence, got approval")
        failures += 1
    elif "Confidence" not in reason_c and "confidence" not in reason_c:
        print(f"  WARN: rejected but reason doesn't mention confidence: {reason_c}")

    # ---------------------------------------------------------------
    print_section("Step 5 — Simulate price movement: take-profit should trigger")
    # ---------------------------------------------------------------
    # Position on MKT-A at entry=0.40, take_profit=0.25 → profit per share needs
    # to be at least 0.25 * 0.40 = $0.10 → new price ≥ 0.50.
    # Move the price to 0.55 to ensure trigger.
    print("  Moving MKT-A price 0.40 -> 0.55 (YES holder profit ~37%)")
    client.move_price("MKT-A", 0.55)

    exits = await executor.check_exits()
    print(f"  executor.check_exits() -> {exits} exit(s)")

    positions = await db.get_open_positions()
    print(f"  db.get_open_positions() after exits -> {len(positions)} position(s)")
    if positions:
        print("  FAIL: expected position to be closed after take-profit")
        failures += 1

    history = await db.get_trade_history(limit=10)
    print(f"  db.get_trade_history() -> {len(history)} closed trade(s)")
    for t in history:
        print(f"    {t['trade_id']}: outcome={t['outcome']} pnl=${t['pnl']:+.2f} "
              f"({t['pnl_pct']:+.1%}) entry=${t['entry_price']:.3f} exit=${t['exit_price']:.3f}")

    # ---------------------------------------------------------------
    print_section("Step 6 — Stop-loss path")
    # ---------------------------------------------------------------
    # Open a new YES position on MKT-D @ 0.50, then crash price to 0.42 (16% loss)
    mkt_d = make_market("MKT-D", "Mock Market D (stop-loss test)", yes_price=0.50)
    await register_market(db, client, mkt_d)
    pred_d = make_prediction("MKT-D", probability=0.70, confidence=0.80, market_price=0.50)
    signal_d, _ = await risk_mgr.approve_trade(
        mkt_d, pred_d, await executor.get_portfolio_state(initial_capital=1000.0)
    )
    if signal_d is None:
        print("  FAIL: setup step failed — could not open stop-loss test position")
        failures += 1
    else:
        await executor.execute(signal_d)
        print("  Position opened on MKT-D at 0.50. Crashing price to 0.42...")
        client.move_price("MKT-D", 0.42)
        exits = await executor.check_exits()
        print(f"  executor.check_exits() -> {exits} exit(s)")
        history = await db.get_trade_history(limit=10)
        latest = history[0] if history else None
        if latest and latest["market_id"] == "MKT-D":
            fc = latest.get("failure_class") or ""
            print(f"  latest trade: outcome={latest['outcome']} "
                  f"failure_class={fc!r} pnl=${latest['pnl']:+.2f}")
            if fc != "stop_loss":
                print(f"  FAIL: expected failure_class=stop_loss, got {fc!r}")
                failures += 1

    # ---------------------------------------------------------------
    print_section("Step 7 — Final portfolio snapshot")
    # ---------------------------------------------------------------
    portfolio = await executor.get_portfolio_state(initial_capital=1000.0)
    print(f"  total_value=${portfolio.total_value:.2f} cash=${portfolio.cash:.2f}")
    print(f"  daily_pnl=${portfolio.daily_pnl:+.2f} open_positions={len(portfolio.positions)}")
    win_rate = await db.get_win_rate()
    print(f"  win_rate={win_rate:.1%}")

    await db.close()

    print_section("SUMMARY")
    if failures == 0:
        print("  All pipeline stages verified. Zero LLM cost.")
        print("  Safe to top up OpenRouter credits and run live.")
        return 0
    else:
        print(f"  {failures} check(s) failed — investigate before going live.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
