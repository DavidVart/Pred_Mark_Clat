"""Tests for the fee model used in net-edge calculation."""

from __future__ import annotations

from src.pipeline.fees import (
    estimate_round_trip_fee,
    kalshi_taker_fee_per_contract,
    net_edge,
    polymarket_taker_fee_pct,
)


class TestPolymarketFees:
    def test_sports_is_cheapest(self):
        assert polymarket_taker_fee_pct("sports") == 0.0075

    def test_crypto_is_most_expensive(self):
        assert polymarket_taker_fee_pct("crypto") == 0.0180

    def test_unknown_category_uses_default(self):
        # Anything unknown falls to the midpoint default
        assert polymarket_taker_fee_pct("foo-bar") > 0.0075
        assert polymarket_taker_fee_pct("foo-bar") < 0.0180

    def test_case_insensitive(self):
        assert polymarket_taker_fee_pct("SPORTS") == 0.0075
        assert polymarket_taker_fee_pct("Sports") == 0.0075

    def test_empty_category(self):
        # Empty string uses default, doesn't crash
        assert polymarket_taker_fee_pct("") > 0


class TestKalshiFees:
    def test_fee_at_50_cents(self):
        # ceil(7 * 0.5 * 0.5) = ceil(1.75) = 2 cents per contract = $0.02
        assert kalshi_taker_fee_per_contract(0.50) == 0.02

    def test_fee_at_edges_is_zero(self):
        assert kalshi_taker_fee_per_contract(0.0) == 0.0
        assert kalshi_taker_fee_per_contract(1.0) == 0.0

    def test_fee_symmetric_around_half(self):
        # Formula is P*(1-P) which is symmetric
        assert kalshi_taker_fee_per_contract(0.30) == kalshi_taker_fee_per_contract(0.70)

    def test_fee_higher_near_midpoint(self):
        # ceil(7 * 0.5 * 0.5) = 2, ceil(7 * 0.1 * 0.9) = ceil(0.63) = 1
        assert kalshi_taker_fee_per_contract(0.50) > kalshi_taker_fee_per_contract(0.10)


class TestRoundTripFees:
    def test_polymarket_taker_round_trip(self):
        fees = estimate_round_trip_fee("polymarket", "sports", 0.5, is_maker_entry=False)
        # 0.75% taker in, 0.75% taker out if not holding to resolution — but default holds to res
        # settles_at_resolution default is False here, so both fees apply
        assert fees.round_trip_pct == 0.0150

    def test_polymarket_maker_entry_saves_half(self):
        taker = estimate_round_trip_fee("polymarket", "sports", 0.5, is_maker_entry=False)
        maker = estimate_round_trip_fee("polymarket", "sports", 0.5, is_maker_entry=True)
        # Maker has 0% entry, only exit fee — saves exactly the entry fee
        assert maker.round_trip_pct < taker.round_trip_pct
        assert abs(taker.round_trip_pct - maker.round_trip_pct - 0.0075) < 1e-9

    def test_hold_to_resolution_waives_exit_fee(self):
        no_exit = estimate_round_trip_fee(
            "polymarket", "sports", 0.5, is_maker_entry=False, settles_at_resolution=True
        )
        with_exit = estimate_round_trip_fee(
            "polymarket", "sports", 0.5, is_maker_entry=False, settles_at_resolution=False
        )
        assert no_exit.round_trip_pct < with_exit.round_trip_pct

    def test_kalshi_high_price_lower_fee_pct(self):
        # Kalshi fee is in cents per contract. At higher prices, fee is a smaller
        # fraction of notional.
        low = estimate_round_trip_fee("kalshi", "sports", 0.10, is_maker_entry=False)
        high = estimate_round_trip_fee("kalshi", "sports", 0.90, is_maker_entry=False)
        assert high.entry_fee_pct < low.entry_fee_pct

    def test_unknown_platform_conservative_default(self):
        fees = estimate_round_trip_fee("some_new_exchange", "sports", 0.5)
        assert fees.round_trip_pct >= 0.03  # Should be pessimistic


class TestNetEdge:
    def test_fees_reduce_edge(self):
        net, fees = net_edge(0.10, "polymarket", "sports", 0.5, is_maker_entry=False)
        assert net < 0.10
        assert net == 0.10 - fees.round_trip_pct

    def test_maker_keeps_more_edge(self):
        taker_net, _ = net_edge(0.10, "polymarket", "sports", 0.5, is_maker_entry=False)
        maker_net, _ = net_edge(0.10, "polymarket", "sports", 0.5, is_maker_entry=True)
        assert maker_net > taker_net

    def test_small_edge_goes_negative_with_fees(self):
        # 2% gross edge on crypto (1.80% taker each way) becomes negative net
        net, _ = net_edge(0.02, "polymarket", "crypto", 0.5,
                          is_maker_entry=False, settles_at_resolution=False)
        assert net < 0
