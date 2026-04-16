"""Tests for the NewsAlpha fair-value math."""

from __future__ import annotations

import math

from src.newsalpha.fair_value import (
    DEFAULT_SIGMA_5MIN,
    FairValueParams,
    _normal_cdf,
    fair_no_probability,
    fair_yes_probability,
)


class TestNormalCDF:
    def test_zero_is_half(self):
        assert abs(_normal_cdf(0.0) - 0.5) < 1e-9

    def test_plus_two_sigma(self):
        # Φ(2) ≈ 0.9772
        assert 0.976 < _normal_cdf(2.0) < 0.978

    def test_minus_two_sigma(self):
        assert 0.022 < _normal_cdf(-2.0) < 0.024

    def test_symmetry(self):
        for x in [0.5, 1.0, 1.5, 3.0]:
            assert abs(_normal_cdf(x) + _normal_cdf(-x) - 1.0) < 1e-9


class TestFairYesProbability:
    def test_spot_equal_strike_gives_half(self):
        """At-the-money with nonzero vol → 50/50."""
        p = fair_yes_probability(spot=100_000, strike=100_000, seconds_remaining=150.0)
        assert abs(p - 0.5) < 1e-9

    def test_spot_above_strike_higher_than_half(self):
        """If we're already above strike with time remaining, YES should be >50%."""
        p = fair_yes_probability(spot=100_300, strike=100_000, seconds_remaining=150.0)
        assert p > 0.5
        assert p < 1.0

    def test_spot_below_strike_lower_than_half(self):
        p = fair_yes_probability(spot=99_700, strike=100_000, seconds_remaining=150.0)
        assert p < 0.5
        assert p > 0.0

    def test_more_time_pulls_toward_half(self):
        """Longer time remaining + same price diff → probability closer to 50%."""
        short = fair_yes_probability(spot=100_300, strike=100_000, seconds_remaining=30.0)
        long = fair_yes_probability(spot=100_300, strike=100_000, seconds_remaining=290.0)
        assert short > long > 0.5

    def test_at_resolution_minute_converges(self):
        """Last 5 seconds: if above strike, should be clamped high."""
        params = FairValueParams(min_seconds_remaining=10.0, clamp_max=0.99)
        p = fair_yes_probability(
            spot=100_100, strike=100_000, seconds_remaining=5.0, params=params
        )
        assert p == 0.99  # clamped

    def test_at_resolution_below_strike_clamps_low(self):
        params = FairValueParams(min_seconds_remaining=10.0, clamp_min=0.01)
        p = fair_yes_probability(
            spot=99_900, strike=100_000, seconds_remaining=5.0, params=params
        )
        assert p == 0.01

    def test_clamping_bounds(self):
        """Very extreme price diffs should clamp at [clamp_min, clamp_max]."""
        p_high = fair_yes_probability(spot=200_000, strike=100_000, seconds_remaining=30.0)
        assert p_high <= 0.99  # default clamp_max

        p_low = fair_yes_probability(spot=50_000, strike=100_000, seconds_remaining=30.0)
        assert p_low >= 0.01

    def test_invalid_inputs_return_half(self):
        assert fair_yes_probability(spot=0, strike=100, seconds_remaining=100) == 0.5
        assert fair_yes_probability(spot=100, strike=0, seconds_remaining=100) == 0.5

    def test_yes_no_sum_to_one(self):
        for spot, strike, sec in [
            (100_000, 100_500, 200),
            (99_500, 100_000, 120),
            (100_200, 100_000, 45),
        ]:
            yes = fair_yes_probability(spot, strike, sec)
            no = fair_no_probability(spot, strike, sec)
            assert abs(yes + no - 1.0) < 1e-9

    def test_custom_sigma_higher_vol_means_less_certain(self):
        """Higher assumed vol → fair probability closer to 0.5."""
        low_vol = FairValueParams(sigma_5min=0.001)   # 0.1%
        high_vol = FairValueParams(sigma_5min=0.01)   # 1%
        # 0.2% above strike, halfway through window
        p_low = fair_yes_probability(100_200, 100_000, 150, params=low_vol)
        p_high = fair_yes_probability(100_200, 100_000, 150, params=high_vol)
        # Low-vol assumption says "big move already happened" → high prob
        # High-vol assumption says "moves this big are noise" → closer to 0.5
        assert p_low > p_high > 0.5
