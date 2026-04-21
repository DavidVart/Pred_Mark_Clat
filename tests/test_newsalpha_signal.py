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


class TestPriceRangeFilter:
    """REGRESSION: do not emit signals on deep-OTM/ITM markets where tick-size
    pathology makes trades untradable."""

    def test_rejects_yes_price_below_5c(self):
        q = _quote(yes=0.02, ref=100_000, seconds_left=120)
        # Market is ultra-low, no trade regardless of fair value
        s = detect_divergence(q, spot=95_000, config=SignalConfig(min_edge=0.01))
        assert s is None

    def test_rejects_yes_price_above_95c(self):
        q = _quote(yes=0.98, ref=100_000, seconds_left=120)
        s = detect_divergence(q, spot=102_000, config=SignalConfig(min_edge=0.01))
        assert s is None

    def test_accepts_yes_price_in_band(self):
        q = _quote(yes=0.40, ref=100_000, seconds_left=60)
        # Spot well above strike → high fair YES, buy YES
        s = detect_divergence(q, spot=100_500, config=SignalConfig(min_edge=0.03))
        # Signal may or may not fire based on math; key is it wasn't price-filtered
        # We verify the opposite side check too: if fair_yes is very high and yes
        # market is 0.40, edge = high, should fire.
        assert s is not None


class TestStrikeDirection:
    """REGRESSION: 'below $X' markets must INVERT the fair-value formula."""

    def test_below_market_inverts_fair_yes(self):
        """A 'below $68k' market: YES wins if BTC < $68k. Fair YES = 1 - P(above)."""
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        # Strike $68k, current BTC $75k → P(above) ≈ 1, so for "below" market fair YES ≈ 0
        q = MarketQuote(
            market_id="below-68k",
            title="Will BTC be below $68,000 on April 21?",
            yes_price=0.40,  # market says 40% (wrong! correct is near 0)
            no_price=0.60,
            window_start=now - timedelta(seconds=60),
            window_end=now + timedelta(seconds=120),
            starting_ref_price=68_000,
            market_type="fixed_strike",
            strike_direction="below",
        )
        s = detect_divergence(q, spot=75_000, config=SignalConfig(min_edge=0.03))
        assert s is not None
        # We buy NO (market overpriced YES). Fair NO should be near 1.0
        # because BTC is way above strike for a "below" market → NO wins.
        assert s.side == "no"
        assert s.fair_value > 0.90  # fair of NO side
        assert s.market_price == 0.60  # NO market price


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
