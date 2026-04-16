"""Tests for the SignalGate dedup logic."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.newsalpha.models import DivergenceSignal
from src.newsalpha.signal import SignalGate


def _signal(market="m1", side="yes", edge=0.10, ts: datetime | None = None) -> DivergenceSignal:
    return DivergenceSignal(
        market_id=market,
        title=f"mock {market}",
        side=side,
        market_price=0.40,
        fair_value=0.40 + edge if side == "yes" else 0.40 - edge,
        edge=edge,
        seconds_remaining=120.0,
        spot_reference=100_000,
        spot_at_window_start=100_000,
        timestamp=ts or datetime.utcnow(),
    )


class TestSignalGate:
    def test_first_signal_passes(self):
        gate = SignalGate(cooldown_seconds=60.0)
        assert gate.should_emit(_signal())

    def test_second_signal_same_market_suppressed_within_cooldown(self):
        gate = SignalGate(cooldown_seconds=60.0)
        t0 = datetime.utcnow()
        assert gate.should_emit(_signal(ts=t0))
        assert not gate.should_emit(_signal(ts=t0 + timedelta(seconds=30)))

    def test_signal_after_cooldown_passes(self):
        gate = SignalGate(cooldown_seconds=60.0)
        t0 = datetime.utcnow()
        assert gate.should_emit(_signal(ts=t0))
        assert gate.should_emit(_signal(ts=t0 + timedelta(seconds=90)))

    def test_side_flip_always_passes(self):
        gate = SignalGate(cooldown_seconds=600.0)
        t0 = datetime.utcnow()
        assert gate.should_emit(_signal(side="yes", ts=t0))
        # Flip side 30s later — should pass despite cooldown
        assert gate.should_emit(_signal(side="no", ts=t0 + timedelta(seconds=30)))

    def test_edge_growth_passes(self):
        gate = SignalGate(cooldown_seconds=600.0, edge_jump_threshold=0.05)
        t0 = datetime.utcnow()
        assert gate.should_emit(_signal(edge=0.10, ts=t0))
        # Edge grows +0.06 → passes even inside cooldown
        assert gate.should_emit(_signal(edge=0.16, ts=t0 + timedelta(seconds=30)))

    def test_edge_shrinkage_suppressed(self):
        gate = SignalGate(cooldown_seconds=600.0, edge_jump_threshold=0.05)
        t0 = datetime.utcnow()
        assert gate.should_emit(_signal(edge=0.15, ts=t0))
        # Edge shrinks to 0.10 → suppressed
        assert not gate.should_emit(_signal(edge=0.10, ts=t0 + timedelta(seconds=30)))

    def test_different_markets_independent(self):
        gate = SignalGate(cooldown_seconds=600.0)
        t0 = datetime.utcnow()
        assert gate.should_emit(_signal(market="m1", ts=t0))
        # Different market — passes immediately
        assert gate.should_emit(_signal(market="m2", ts=t0 + timedelta(seconds=10)))

    def test_reset_clears_state(self):
        gate = SignalGate(cooldown_seconds=600.0)
        t0 = datetime.utcnow()
        assert gate.should_emit(_signal(market="m1", ts=t0))
        assert not gate.should_emit(_signal(market="m1", ts=t0 + timedelta(seconds=30)))
        gate.reset("m1")
        assert gate.should_emit(_signal(market="m1", ts=t0 + timedelta(seconds=30)))

    def test_reset_all(self):
        gate = SignalGate(cooldown_seconds=600.0)
        t0 = datetime.utcnow()
        gate.should_emit(_signal(market="a", ts=t0))
        gate.should_emit(_signal(market="b", ts=t0))
        gate.reset()
        assert gate.should_emit(_signal(market="a", ts=t0 + timedelta(seconds=1)))
        assert gate.should_emit(_signal(market="b", ts=t0 + timedelta(seconds=1)))
