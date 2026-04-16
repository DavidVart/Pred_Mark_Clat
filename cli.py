"""Unified CLI for the prediction market trading bot."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer

from config.settings import load_settings
from src.utils.logging import setup_logging, get_logger

app = typer.Typer(
    name="pmc",
    help="AI-Powered Prediction Market Trading Bot for Polymarket & Kalshi",
)


@app.command()
def health():
    """Check API keys, connectivity, database, and Python version."""
    import sys
    from pathlib import Path

    settings = load_settings()
    setup_logging(settings.log_level)

    typer.echo("=== Prediction Market Bot — Health Check ===\n")
    typer.echo(f"Python: {sys.version}")
    typer.echo(f"Trading Mode: {settings.trading_mode}")
    typer.echo(f"Database: {settings.db_path}")
    typer.echo()

    # Check API keys
    checks = {
        "OPENROUTER_API_KEY": bool(settings.openrouter_api_key),
        "POLYGON_WALLET_PRIVATE_KEY": bool(settings.polymarket.wallet_private_key),
        "KALSHI_API_KEY": bool(settings.kalshi.api_key),
        "KALSHI_PRIVATE_KEY": Path(settings.kalshi.private_key_path).exists() if settings.kalshi.private_key_path else False,
    }

    for name, ok in checks.items():
        status = typer.style("OK", fg=typer.colors.GREEN) if ok else typer.style("MISSING", fg=typer.colors.RED)
        typer.echo(f"  {name}: {status}")

    typer.echo()

    # Risk parameters
    typer.echo("Risk Parameters:")
    typer.echo(f"  Kelly Fraction: {settings.risk.kelly_fraction}")
    typer.echo(f"  Max Position Size: {settings.risk.max_position_size_pct:.0%}")
    typer.echo(f"  Max Positions: {settings.risk.max_positions}")
    typer.echo(f"  Min Edge: {settings.risk.min_edge:.0%}")
    typer.echo(f"  Min Confidence: {settings.risk.min_confidence}")
    typer.echo(f"  Max Daily Loss: {settings.risk.max_daily_loss_pct:.0%}")
    typer.echo(f"  Max Drawdown: {settings.risk.max_drawdown_pct:.0%}")
    typer.echo(f"  Daily AI Cost Limit: ${settings.cost.daily_ai_cost_limit:.2f}")
    typer.echo()

    # DB connectivity
    async def check_db():
        from src.db.manager import DatabaseManager
        db = DatabaseManager(settings.db_path)
        try:
            await db.initialize()
            typer.echo(typer.style("Database: OK", fg=typer.colors.GREEN))
        except Exception as e:
            typer.echo(typer.style(f"Database: FAILED — {e}", fg=typer.colors.RED))
        finally:
            await db.close()

    asyncio.run(check_db())


@app.command()
def run(
    paper: bool = typer.Option(True, "--paper/--live", help="Paper trading mode"),
    platforms: str = typer.Option("both", help="Platforms: both, polymarket, kalshi"),
    interval: int = typer.Option(5, help="Minutes between scan cycles"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Start the trading bot."""
    settings = load_settings()
    setup_logging(log_level)
    logger = get_logger("cli")

    mode = "PAPER" if paper else "LIVE"
    logger.info("bot_starting", mode=mode, platforms=platforms, interval=interval)
    typer.echo(f"Starting bot in {mode} mode — platforms={platforms}, interval={interval}m")

    if not paper:
        typer.confirm(
            "WARNING: You are about to start LIVE trading with real money. Continue?",
            abort=True,
        )

    from src.pipeline.orchestrator import run_bot
    asyncio.run(run_bot(settings, paper_mode=paper, platforms=platforms, interval=interval))


@app.command()
def status():
    """Show portfolio, open positions, and daily PnL."""
    settings = load_settings()

    async def show_status():
        from src.db.manager import DatabaseManager
        db = DatabaseManager(settings.db_path)
        await db.initialize()

        positions = await db.get_open_positions()
        daily_pnl = await db.get_daily_pnl()
        daily_cost = await db.get_daily_ai_cost()
        win_rate = await db.get_win_rate()
        ks = await db.get_kill_switch()

        typer.echo("=== Portfolio Status ===\n")
        typer.echo(f"Kill Switch: {'ACTIVE — ' + ks.get('reason', '') if ks.get('active') else 'OFF'}")
        typer.echo(f"Open Positions: {len(positions)}")
        typer.echo(f"Daily PnL: ${daily_pnl:+.2f}")
        typer.echo(f"Daily AI Cost: ${daily_cost:.2f}")
        typer.echo(f"Win Rate: {win_rate:.1%}")

        if positions:
            typer.echo("\n  Positions:")
            for p in positions:
                side = p["side"].upper()
                entry = p["entry_price"]
                current = p["current_price"]
                pnl = (current - entry) * p["quantity"] if p["side"] == "yes" else (entry - current) * p["quantity"]
                typer.echo(f"    [{p['platform']}] {p['title'][:50]} | {side} @ {entry:.2f} → {current:.2f} | PnL: ${pnl:+.2f}")

        await db.close()

    asyncio.run(show_status())


@app.command()
def history(limit: int = typer.Option(20, help="Number of trades to show")):
    """Show trade history."""
    settings = load_settings()

    async def show_history():
        from src.db.manager import DatabaseManager
        db = DatabaseManager(settings.db_path)
        await db.initialize()

        trades = await db.get_trade_history(limit)
        if not trades:
            typer.echo("No trade history yet.")
            await db.close()
            return

        typer.echo(f"=== Last {len(trades)} Trades ===\n")
        for t in trades:
            outcome = "W" if t["pnl"] > 0 else "L"
            typer.echo(
                f"  [{outcome}] {t['title'][:45]} | {t['side'].upper()} | "
                f"PnL: ${t['pnl']:+.2f} ({t['pnl_pct']:+.1%}) | "
                f"{t['hold_duration_hours']:.1f}h | {t['platform']}"
            )
        await db.close()

    asyncio.run(show_history())


@app.command()
def arb(
    paper: bool = typer.Option(True, "--paper/--live", help="Paper trading mode"),
    interval: int = typer.Option(30, help="Seconds between scans (prediction market arb is not HFT — 30s is fine)"),
    min_spread: float = typer.Option(0.005, help="Minimum net spread to execute (0.005 = 0.5%)"),
    notional: float = typer.Option(100.0, help="Max dollars per two-leg trade"),
    seed_file: Optional[str] = typer.Option(None, help="Path to pairs JSON (defaults to configs/market_pairs.json)"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run the cross-platform arbitrage bot (Polymarket <-> Kalshi)."""
    settings = load_settings()
    setup_logging(log_level)
    logger = get_logger("cli")

    mode = "PAPER" if paper else "LIVE"
    logger.info(
        "arb_cli_start",
        mode=mode,
        interval=interval,
        min_spread=min_spread,
        notional=notional,
    )
    typer.echo(f"Starting ARB bot in {mode} mode — interval={interval}s min_spread={min_spread} notional=${notional}")

    if not paper:
        typer.confirm(
            "WARNING: Cross-platform arb with real money. Both legs are real orders. "
            "If market pair is misaligned, both can lose. Continue?",
            abort=True,
        )

    from src.arbitrage.orchestrator import run_arb
    asyncio.run(run_arb(
        settings,
        paper_mode=paper,
        interval_seconds=interval,
        min_net_spread=min_spread,
        max_notional=notional,
        seed_file=seed_file,
    ))


@app.command()
def arb_status():
    """Show open arb positions and expected profit."""
    settings = load_settings()

    async def show():
        from src.db.manager import DatabaseManager
        db = DatabaseManager(settings.db_path)
        await db.initialize()

        try:
            cursor = await db.db.execute(
                "SELECT * FROM arb_positions ORDER BY opened_at DESC LIMIT 50"
            )
            rows = await cursor.fetchall()
        except Exception:
            typer.echo("No arb_positions table yet — run `python cli.py arb --paper` first.")
            await db.close()
            return

        if not rows:
            typer.echo("No arb positions yet.")
            await db.close()
            return

        typer.echo("=== Arb Positions ===\n")
        total_expected = 0.0
        total_realized = 0.0
        for r in rows:
            r = dict(r)
            status_str = "OPEN" if r.get("closed_at") is None else ("WIN" if (r.get("realized_pnl") or 0) > 0 else "LOSS")
            expected = r.get("expected_profit") or 0.0
            realized = r.get("realized_pnl") or 0.0
            total_expected += expected
            if r.get("closed_at"):
                total_realized += realized
            typer.echo(
                f"  [{status_str}] {r['pair_id'][:30]} | "
                f"poly {r['poly_side']}@{r['poly_entry_price']:.3f} + "
                f"kalshi {r['kalshi_side']}@{r['kalshi_entry_price']:.3f} | "
                f"basket=${r['basket_cost']:.3f} | "
                f"expected=${expected:+.2f} realized=${realized:+.2f}"
            )
        typer.echo()
        typer.echo(f"Total expected profit (all arbs): ${total_expected:+.2f}")
        typer.echo(f"Total realized profit (closed):   ${total_realized:+.2f}")
        await db.close()

    asyncio.run(show())


@app.command(name="news-scan")
def news_scan(
    db: str = typer.Option("newsalpha.db", help="Path to NewsAlpha DB"),
    poll: int = typer.Option(3, help="Seconds between Polymarket market refreshes"),
    min_edge: float = typer.Option(0.03, help="Minimum fair-vs-market gap to log a signal"),
    observe: bool = typer.Option(False, "--observe/--trade", help="Observe-only (no paper trades) vs trade mode"),
    bankroll: float = typer.Option(1000.0, help="Starting paper bankroll"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run NewsAlpha — divergence scanner + paper trading engine."""
    logger = get_logger("cli")
    mode = "OBSERVE" if observe else "PAPER TRADE"
    logger.info("newsalpha_cli_start", poll=poll, min_edge=min_edge, mode=mode)
    typer.echo(f"NewsAlpha {mode} — poll={poll}s min_edge={min_edge} db={db} bankroll=${bankroll}")

    from src.newsalpha.orchestrator import run_newsalpha
    asyncio.run(run_newsalpha(
        db_path=db,
        poll_seconds=poll,
        log_level=log_level,
        min_edge=min_edge,
        observe_only=observe,
        bankroll=bankroll,
    ))


@app.command(name="news-status")
def news_status(db: str = typer.Option("newsalpha.db", help="Path to NewsAlpha DB")):
    """Show NewsAlpha signals, open positions, and trade history."""
    async def show():
        from src.newsalpha.db import NewsAlphaDB
        manager = NewsAlphaDB(db)
        await manager.initialize()
        try:
            signals = await manager.recent_signals(10)
            today = await manager.count_signals_today()

            # Trades
            cursor = await manager.db.execute(
                "SELECT * FROM na_trades ORDER BY closed_at DESC LIMIT 20"
            )
            trades = [dict(r) for r in await cursor.fetchall()]

            # Open positions
            cursor2 = await manager.db.execute("SELECT * FROM na_positions")
            positions = [dict(r) for r in await cursor2.fetchall()]

            # Stats
            cursor3 = await manager.db.execute(
                "SELECT COUNT(*) as cnt, SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins, "
                "SUM(pnl) as total_pnl FROM na_trades"
            )
            stats = dict(await cursor3.fetchone())

            typer.echo("=== NewsAlpha Status ===\n")
            typer.echo(f"Signals today: {today}")

            # Trade summary
            cnt = stats.get("cnt") or 0
            wins = stats.get("wins") or 0
            total_pnl = stats.get("total_pnl") or 0.0
            win_rate = (wins / cnt * 100) if cnt > 0 else 0.0
            pnl_color = typer.colors.GREEN if total_pnl >= 0 else typer.colors.RED
            typer.echo(f"Total trades: {cnt} | Wins: {wins} | Win rate: {win_rate:.1f}%")
            typer.echo(typer.style(f"Total PnL: ${total_pnl:+.2f}", fg=pnl_color))

            # Open positions
            if positions:
                typer.echo(f"\nOpen positions ({len(positions)}):")
                for p in positions:
                    typer.echo(
                        f"  {p['position_id']}: {p['title'][:40]} | "
                        f"{p['side'].upper()} @ {p['entry_price']:.3f} | "
                        f"cost=${p['cost_basis']:.2f} | edge={p.get('signal_edge', 0):.3f}"
                    )

            # Recent trades
            if trades:
                typer.echo(f"\nRecent trades (last {len(trades)}):")
                for t in trades:
                    outcome_icon = "W" if t["outcome"] == "win" else "L"
                    pnl_c = typer.colors.GREEN if t["pnl"] > 0 else typer.colors.RED
                    typer.echo(
                        f"  [{outcome_icon}] {t['title'][:35]} | {t['side'].upper()} | "
                        f"PnL: " + typer.style(f"${t['pnl']:+.2f} ({t['pnl_pct']:+.1%})", fg=pnl_c) +
                        f" | {t['hold_seconds']:.0f}s | {t['exit_reason']}"
                    )

            # Recent signals
            if signals:
                typer.echo(f"\nRecent signals (last {len(signals)}):")
                for s in signals:
                    typer.echo(
                        f"  [{s['timestamp'][:19]}] {s['title'][:40]} | "
                        f"{s['side'].upper()} @ {s['market_price']:.3f} "
                        f"(fair {s['fair_value']:.3f}, edge {s['edge']:+.3f})"
                    )
        finally:
            await manager.close()

    asyncio.run(show())


@app.command()
def kill(reason: str = typer.Argument("Manual kill")):
    """Activate the kill switch — halts all new trades."""
    settings = load_settings()

    async def do_kill():
        from src.db.manager import DatabaseManager
        db = DatabaseManager(settings.db_path)
        await db.initialize()
        await db.set_kill_switch(True, reason, "cli")
        await db.close()

    asyncio.run(do_kill())
    typer.echo(typer.style(f"Kill switch ACTIVATED: {reason}", fg=typer.colors.RED))


@app.command()
def unkill():
    """Deactivate the kill switch — resume trading."""
    settings = load_settings()

    async def do_unkill():
        from src.db.manager import DatabaseManager
        from src.utils.kill_switch import deactivate
        db = DatabaseManager(settings.db_path)
        await db.initialize()
        await deactivate(db)
        await db.close()

    asyncio.run(do_unkill())
    typer.echo(typer.style("Kill switch DEACTIVATED", fg=typer.colors.GREEN))


if __name__ == "__main__":
    app()
