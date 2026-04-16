"""Tests for the NewsAlpha divergence signal detector."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.newsalpha.models import MarketQuote
from src.newsalpha.signal import SignalConfig, detect_divergence


def _quote(yes=0.50, no=None, ref=100_000, seconds_left=180.0, window_total=300.0) -> MarketQuote:
    """Build a mock Polymarket 5M BTC quote."""
    no = 1.0 - yes if no is None else no
    now = datetime.utcnow()
    end = now + timedelta(seconds=seconds_left)
    start = end - timedelta(seconds=window_total)
    return MarketQuote(
        market_id="mock-btc-5m",
        title="Bitcoin Up or Down - 5M #1",
        yes_price=yes,
        no_price=no,
        window_start=start,
        window_end=end,
        starting_ref_price=ref,
    )


class TestDetectDivergence:
    def test_no_signal_when_aligned(self):
        """Spot at strike, market at 0.50 → no divergence."""
        q = _quote(yes=0.50, ref=100_000)
        s = detect_divergence(q, spot=100_000, config=SignalConfig(min_edge=0.03))
        assert s is None

    def test_signal_when_spot_way_above_strike_and_market_cheap(self):
        """BTC clearly above strike + market still at 0.50 → BUY YES."""
        q = _quote(yes=0.50, ref=100_000, seconds_left=60.0)  # short time, big move
        s = detect_divergence(q, spot=100_400, config=SignalConfig(min_edge=0.03))
        assert s is not None
        assert s.side == "yes"
        assert s.edge > 0.03

    def test_signal_when_spot_way_below_strike_and_market_expensive(self):
        """BTC below strike + YES market still at 0.55 → fair YES is low → BUY NO."""
        q = _quote(yes=0.55, ref=100_000, seconds_left=60.0)
        s = detect_divergence(q, spot=99_600, config=SignalConfig(min_edge=0.03))
        assert s is not None
        assert s.side == "no"
        assert s.edge > 0.03

    def test_skips_too_close_to_resolution(self):
        """Below min_seconds_remaining → return None to avoid stale quotes."""
        q = _quote(yes=0.50, ref=100_000, seconds_left=10.0)
        cfg = SignalConfig(min_edge=0.01, min_seconds_remaining=30.0)
        s = detect_divergence(q, spot=100_400, config=cfg)
        assert s is None

    def test_skips_too_far_from_resolution(self):
        """Beyond max_seconds_remaining → return None (far-out noise)."""
        q = _quote(yes=0.50, ref=100_000, seconds_left=900.0)
        cfg = SignalConfig(min_edge=0.01, max_seconds_remaining=600.0)
        s = detect_divergence(q, spot=101_000, config=cfg)
        assert s is None

    def test_sub_threshold_edge_returns_none(self):
        """Small divergence that doesn't clear min_edge → None."""
        q = _quote(yes=0.50, ref=100_000, seconds_left=150.0)
        s = detect_divergence(q, spot=100_030, config=SignalConfig(min_edge=0.10))
        assert s is None

    def test_uses_fallback_ref_when_missing(self):
        """If market doesn't expose starting_ref_price, use spot."""
        now = datetime.utcnow()
        q = MarketQuote(
            market_id="no-ref-market",
            title="BTC higher at 3pm",
            yes_price=0.30,
            no_price=0.70,
            window_start=now - timedelta(seconds=100),
            window_end=now + timedelta(seconds=60),
            starting_ref_price=None,
        )
        # With ref = spot, strike = spot, fair YES = 0.5, market = 0.30 → edge 0.20 → BUY YES
        s = detect_divergence(q, spot=100_000)
        assert s is not None
        assert s.side == "yes"
        assert s.edge > 0.15

    def test_signal_records_correct_prices(self):
        """Returned signal has correct market/fair prices for the chosen side."""
        q = _quote(yes=0.40, ref=100_000, seconds_left=60.0)
        s = detect_divergence(q, spot=100_500, config=SignalConfig(min_edge=0.03))
        assert s is not None
        assert s.side == "yes"
        assert s.market_price == 0.40
        # Fair yes should be > 0.5 because spot > strike with little time left
        assert s.fair_value > 0.5
