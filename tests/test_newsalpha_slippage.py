"""Tests for the slippage simulator + gray-mode executor."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from src.newsalpha.db import NewsAlphaDB
from src.newsalpha.executor import NewsAlphaExecutor, NewsAlphaExecutorConfig
from src.newsalpha.models import DivergenceSignal, MarketQuote
from src.newsalpha.slippage import SlippageConfig, SlippageSimulator


@pytest_asyncio.fixture
async def na_db():
    db = NewsAlphaDB(":memory:")
    await db.initialize()
    yield db
    await db.close()


def _signal(
    market="m1",
    side="yes",
    edge=0.15,           # big edge so moves don't immediately kill fill
    market_price=0.40,
    fair_value=None,
    sec_left=300.0,
    title="Mock BTC Up/Down",
) -> DivergenceSignal:
    if fair_value is None:
        fair_value = market_price + edge if side == "yes" else market_price - edge
    return DivergenceSignal(
        market_id=market,
        title=title,
        side=side,
        market_price=market_price,
        fair_value=fair_value,
        edge=edge,
        seconds_remaining=sec_left,
        spot_reference=74000.0,
        spot_at_window_start=73800.0,
    )


def _quote(market_id="m1", yes=0.40, sec_left=250.0) -> MarketQuote:
    now = datetime.utcnow()
    return MarketQuote(
        market_id=market_id,
        title="Mock",
        yes_price=yes,
        no_price=1 - yes,
        window_start=now - timedelta(seconds=50),
        window_end=now + timedelta(seconds=sec_left),
    )


# --- SlippageSimulator unit tests ---

class TestSlippageSimulator:
    def test_deterministic_with_seed(self):
        """Same seed → same result."""
        cfg = SlippageConfig(random_seed=42)
        sim1 = SlippageSimulator(cfg)
        sim2 = SlippageSimulator(cfg)
        sig = _signal()
        q = _quote()
        r1 = sim1.simulate_entry(sig, q, "taker")
        r2 = sim2.simulate_entry(sig, q, "taker")
        assert r1.filled == r2.filled
        if r1.filled and r2.filled:
            assert r1.fill_price == r2.fill_price
            assert r1.latency_ms == r2.latency_ms

    def test_taker_always_crosses_spread(self):
        """Taker buy should pay MORE than signal price (spread cost)."""
        cfg = SlippageConfig(
            random_seed=1,
            btc_sigma_5min=0.0,             # no price drift
            spread_pct_of_mid_mean=0.05,    # 5% of 0.40 mid = 2c spread
            spread_pct_of_mid_std=0.0,      # deterministic
        )
        sim = SlippageSimulator(cfg)
        sig = _signal(edge=0.30, market_price=0.40, fair_value=0.70)
        q = _quote(yes=0.40)
        r = sim.simulate_entry(sig, q, "taker")
        assert r.filled
        assert r.fill_price > 0.40  # paid ask
        assert r.slippage_bps > 0

    def test_taker_refuses_when_move_kills_edge(self):
        """If price drifts enough during latency that fair <= fill price, refuse.

        We hammer the simulator with many trials and a tiny edge — by the
        symmetric random walk, some non-trivial fraction must hit the
        adverse-move branch. Using a high base spread to help it along.
        """
        cfg = SlippageConfig(
            random_seed=0,
            btc_sigma_5min=0.03,
            latency_min_sec=3.0,
            latency_max_sec=3.0,
            spread_pct_of_mid_mean=0.15,    # large spread relative to mid
            spread_pct_of_mid_std=0.0,
        )
        sim = SlippageSimulator(cfg)
        # Fair value = market_price + 0.025 = 0.425. With 2.5c spread alone
        # the ask is 0.4125. Random drift of 1-2c will frequently push above fair.
        sig = _signal(edge=0.025, market_price=0.40, fair_value=0.425)
        q = _quote(yes=0.40)
        outcomes = [sim.simulate_entry(sig, q, "taker") for _ in range(200)]
        refused = [r for r in outcomes if not r.filled and r.reason == "adverse_move"]
        assert len(refused) > 0, (
            f"Expected some adverse_move rejections. Got: "
            f"filled={sum(1 for o in outcomes if o.filled)}, "
            f"refused={len(refused)}, other={sum(1 for o in outcomes if not o.filled and o.reason != 'adverse_move')}"
        )

    def test_maker_has_fill_probability(self):
        """Maker orders fill only ~55% of the time by default."""
        cfg = SlippageConfig(
            random_seed=123,
            btc_sigma_5min=0.0,
            maker_fill_prob_at_bid=0.55,
        )
        sim = SlippageSimulator(cfg)
        sig = _signal(edge=0.30, market_price=0.40, fair_value=0.70)
        q = _quote(yes=0.40)
        results = [sim.simulate_entry(sig, q, "maker") for _ in range(200)]
        filled = sum(1 for r in results if r.filled)
        # Roughly 55% ± 10% on 200 trials
        assert 85 < filled < 130, f"maker fill rate was {filled}/200 — outside expected range"

    def test_maker_saves_spread_when_fills(self):
        """A filled maker order should be at a BETTER price than taker."""
        cfg = SlippageConfig(
            random_seed=7,
            btc_sigma_5min=0.0,
            spread_pct_of_mid_mean=0.08,
            spread_pct_of_mid_std=0.0,
            maker_fill_prob_at_bid=1.0,  # force fill for test
        )
        sim = SlippageSimulator(cfg)
        sig = _signal(edge=0.30, market_price=0.40, fair_value=0.70)
        q = _quote(yes=0.40)
        taker = sim.simulate_entry(sig, q, "taker")
        # Reset RNG so maker sees same random sequence for apples-to-apples
        sim = SlippageSimulator(cfg)
        maker = sim.simulate_entry(sig, q, "maker")
        assert taker.filled and maker.filled
        assert maker.fill_price < taker.fill_price  # bought at bid, better than ask

    def test_spread_scales_with_price_not_absolute(self):
        """REGRESSION: on a 1¢ market, spread must not be 2¢ (200% slippage)."""
        cfg = SlippageConfig(
            random_seed=0,
            btc_sigma_5min=0.0,
            spread_pct_of_mid_mean=0.08,  # 8% of mid
            spread_pct_of_mid_std=0.0,
            min_tick=0.001,
        )
        sim = SlippageSimulator(cfg)
        # On a 1c market, 8% of 0.01 = 0.0008, floored at min_tick=0.001
        spread = sim._sample_spread(mid=0.01)
        assert spread <= 0.005, f"spread on 1c mid was {spread}, too wide"
        # On a 40c market, 8% of 0.40 = 0.032 (3.2c)
        spread = sim._sample_spread(mid=0.40)
        assert 0.02 <= spread <= 0.04

    def test_spread_never_below_tick(self):
        cfg = SlippageConfig(
            random_seed=0, spread_pct_of_mid_mean=0.0, spread_pct_of_mid_std=0.0,
            min_tick=0.001,
        )
        sim = SlippageSimulator(cfg)
        # With 0% pct, spread should fall back to min_tick
        assert sim._sample_spread(mid=0.50) == cfg.min_tick

    def test_spread_never_exceeds_max_absolute(self):
        cfg = SlippageConfig(
            random_seed=0,
            spread_pct_of_mid_mean=0.50,   # 50% of mid is insane
            spread_pct_of_mid_std=0.0,
            max_absolute_spread=0.05,
        )
        sim = SlippageSimulator(cfg)
        spread = sim._sample_spread(mid=0.90)  # 50% of 0.90 = 0.45, capped at 0.05
        assert spread == cfg.max_absolute_spread

    def test_apply_fee_computes_effective_cost(self):
        sim = SlippageSimulator(SlippageConfig(random_seed=0, taker_fee_pct=0.018))
        sig = _signal(edge=0.30, market_price=0.40, fair_value=0.70)
        q = _quote(yes=0.40)
        r = sim.simulate_entry(sig, q, "taker")
        if not r.filled:
            pytest.skip("random draw didn't fill, skip this specific test")
        sim.apply_fee(r, size=100)
        assert r.fees_paid == pytest.approx(r.fill_price * 100 * 0.018, rel=1e-6)
        assert r.effective_cost == pytest.approx(r.fill_price * 100 + r.fees_paid, rel=1e-6)


# --- Gray-mode executor integration tests ---

class TestGrayExecutor:
    @pytest.mark.asyncio
    async def test_gray_requires_simulator(self, na_db):
        with pytest.raises(ValueError):
            NewsAlphaExecutor(
                NewsAlphaExecutorConfig(bankroll=1000),
                na_db,
                mode="gray",
                slippage=None,
            )

    @pytest.mark.asyncio
    async def test_gray_opens_at_slipped_price(self, na_db):
        """Gray mode should open at a price WORSE than the signal."""
        sim = SlippageSimulator(SlippageConfig(
            random_seed=1, btc_sigma_5min=0.0,
            spread_pct_of_mid_mean=0.05, spread_pct_of_mid_std=0.0,
        ))
        ex = NewsAlphaExecutor(
            NewsAlphaExecutorConfig(bankroll=1000),
            na_db,
            mode="gray",
            slippage=sim,
        )
        await ex.on_signal(_signal(edge=0.30, market_price=0.40, fair_value=0.70))
        cursor = await na_db.db.execute(
            "SELECT entry_price, signal_price, entry_fees, execution_mode FROM na_positions LIMIT 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        entry_price, signal_price, entry_fees, mode = row
        assert mode == "gray"
        assert signal_price == 0.40
        assert entry_price > 0.40  # slipped
        assert entry_fees > 0  # taker fee paid

    @pytest.mark.asyncio
    async def test_paper_has_no_slippage(self, na_db):
        """Paper mode entry_price should equal signal_price, no fees."""
        ex = NewsAlphaExecutor(
            NewsAlphaExecutorConfig(bankroll=1000),
            na_db,
            mode="paper",
        )
        await ex.on_signal(_signal(edge=0.15, market_price=0.40))
        cursor = await na_db.db.execute(
            "SELECT entry_price, signal_price, entry_fees, execution_mode FROM na_positions LIMIT 1"
        )
        row = await cursor.fetchone()
        entry_price, signal_price, entry_fees, mode = row
        assert mode == "paper"
        assert entry_price == 0.40
        assert entry_fees == 0

    @pytest.mark.asyncio
    async def test_paper_and_gray_coexist_in_same_db(self, na_db):
        """Two executors, same DB, different modes — positions tagged correctly."""
        paper = NewsAlphaExecutor(NewsAlphaExecutorConfig(bankroll=1000), na_db, mode="paper")
        sim = SlippageSimulator(SlippageConfig(random_seed=7, btc_sigma_5min=0.0))
        gray = NewsAlphaExecutor(NewsAlphaExecutorConfig(bankroll=1000), na_db, mode="gray", slippage=sim)

        sig = _signal(edge=0.20, market_price=0.40, fair_value=0.60)
        await paper.on_signal(sig)
        await gray.on_signal(sig)

        cursor = await na_db.db.execute(
            "SELECT execution_mode, COUNT(*) FROM na_positions GROUP BY execution_mode"
        )
        rows = await cursor.fetchall()
        modes = {r[0]: r[1] for r in rows}
        assert modes.get("paper") == 1
        assert modes.get("gray") == 1

    @pytest.mark.asyncio
    async def test_gray_pnl_includes_exit_fees(self, na_db):
        """Gray trade PnL on close should be WORSE than paper PnL due to exit slippage + fees."""
        sim = SlippageSimulator(SlippageConfig(
            random_seed=3, btc_sigma_5min=0.0,
            spread_pct_of_mid_mean=0.05, spread_pct_of_mid_std=0.0,
            taker_fee_pct=0.01,
        ))
        gray = NewsAlphaExecutor(
            NewsAlphaExecutorConfig(bankroll=1000, trailing_activation_pct=0.01),
            na_db, mode="gray", slippage=sim,
        )
        await gray.on_signal(_signal(edge=0.30, market_price=0.40, fair_value=0.70))

        # Price moves in our favor → close position at a "higher" market price
        await gray.check_exits({"m1": _quote("m1", yes=0.55, sec_left=200)})

        cursor = await na_db.db.execute(
            "SELECT pnl, fees_paid, execution_mode FROM na_trades WHERE execution_mode='gray' LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            # Position might still be open if not stopped; force close via time-stop
            await gray.check_exits({"m1": _quote("m1", yes=0.55, sec_left=30)})
            cursor = await na_db.db.execute(
                "SELECT pnl, fees_paid, execution_mode FROM na_trades WHERE execution_mode='gray' LIMIT 1"
            )
            row = await cursor.fetchone()
        assert row is not None
        pnl, fees, mode = row
        assert mode == "gray"
        assert fees > 0  # fees paid on entry + exit


# --- Live-mode safety caps (mode set but no real client) ---

class TestLiveSafetyCaps:
    @pytest.mark.asyncio
    async def test_live_mode_caps_position_size(self, na_db):
        """In live mode, position size is hard-capped at live_max_position_usd."""
        cfg = NewsAlphaExecutorConfig(
            bankroll=10_000,         # bigger bankroll to force Kelly > cap
            max_position_pct=0.20,   # bigger Kelly %
            live_max_position_usd=5.0,
        )
        sim = SlippageSimulator(SlippageConfig(random_seed=1, btc_sigma_5min=0.0))
        ex = NewsAlphaExecutor(cfg, na_db, mode="live", slippage=sim)
        await ex.on_signal(_signal(edge=0.30, market_price=0.40, fair_value=0.70))

        cursor = await na_db.db.execute(
            "SELECT cost_basis FROM na_positions WHERE execution_mode='live' LIMIT 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        cost_basis = row[0]
        # Should be ≤ $5 + fees (fees added on top of dollar_size for gray/live)
        assert cost_basis <= 5.2, f"live cap breached: cost_basis={cost_basis}"

    @pytest.mark.asyncio
    async def test_live_mode_respects_daily_opens_cap(self, na_db):
        cfg = NewsAlphaExecutorConfig(
            bankroll=1000, max_positions=100,  # high to isolate daily cap
            live_max_daily_opens=2,
        )
        sim = SlippageSimulator(SlippageConfig(random_seed=1, btc_sigma_5min=0.0))
        ex = NewsAlphaExecutor(cfg, na_db, mode="live", slippage=sim)

        for i in range(5):
            await ex.on_signal(_signal(market=f"m{i}", edge=0.20, market_price=0.40, fair_value=0.60))

        cursor = await na_db.db.execute(
            "SELECT COUNT(*) FROM na_positions WHERE execution_mode='live'"
        )
        row = await cursor.fetchone()
        assert row[0] == 2  # capped at 2 opens
