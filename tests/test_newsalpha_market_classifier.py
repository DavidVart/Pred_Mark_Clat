"""Tests for market-type classification — uses REAL titles pulled from Polymarket."""

from __future__ import annotations

import pytest

from src.newsalpha.market_classifier import MarketClassification, classify_title


class TestUpOrDown:
    @pytest.mark.parametrize("title", [
        "Bitcoin Up or Down on April 21?",
        "Bitcoin Up or Down - April 21, 8AM ET",
        "Bitcoin Up or Down - April 17, 12:00PM-4:00PM ET",
        "bitcoin up or down",  # case insensitive
    ])
    def test_up_or_down_detected(self, title):
        c = classify_title(title)
        assert c.type == "up_or_down"
        assert c.strike is None


class TestFixedStrike:
    @pytest.mark.parametrize("title,expected_strike,direction", [
        ("Will the price of Bitcoin be above $72,000 on April 21?", 72000, "above"),
        ("Will the price of Bitcoin be above $66,000 on April 21?", 66000, "above"),
        ("Will the price of Bitcoin be above $82,000 on April 21?", 82000, "above"),
        ("Will BTC be above $150,000 in April?", 150000, "above"),
        ("Will Bitcoin reach $150,000 in April?", 150000, "above"),
        ("Will Bitcoin hit $100,000 this year?", 100000, "above"),
        # "k" suffix
        ("Will Bitcoin reach $150k in April?", 150000, "above"),
        ("Will BTC hit 100k?", 100000, "above"),
        # "below"
        ("Will Bitcoin be below $60,000 on April 21?", 60000, "below"),
    ])
    def test_fixed_strike_parsed(self, title, expected_strike, direction):
        c = classify_title(title)
        assert c.type == "fixed_strike", f"Expected fixed_strike for: {title}"
        assert c.strike == expected_strike
        assert c.direction == direction

    def test_fixed_strike_is_supported(self):
        c = classify_title("Will the price of Bitcoin be above $72,000 on April 21?")
        assert c.is_supported


class TestBetween:
    @pytest.mark.parametrize("title,low,high", [
        ("Will the price of Bitcoin be between $74,000 and $76,000 on April 21?", 74000, 76000),
        ("Will the price of Bitcoin be between $66,000 and $68,000 on April 21?", 66000, 68000),
        ("Bitcoin between $80k and $90k", 80000, 90000),
    ])
    def test_between_detected(self, title, low, high):
        c = classify_title(title)
        assert c.type == "between"
        assert c.strike_low == low
        assert c.strike_high == high

    def test_between_not_currently_supported(self):
        """We detect between markets but scanner should skip them in v1."""
        c = classify_title("Will BTC be between $74,000 and $76,000 today?")
        assert not c.is_supported


class TestUnknown:
    @pytest.mark.parametrize("title", [
        "Will there be a Bitcoin ETF approval by April?",
        "Something about crypto without numbers",
        "Ethereum stuff",  # not BTC
        "Will a country buy Bitcoin in 2026?",
    ])
    def test_unclassifiable_returns_unknown(self, title):
        c = classify_title(title)
        assert c.type == "unknown"
        assert not c.is_supported


class TestEdgeCases:
    def test_malformed_dollar_amount(self):
        c = classify_title("Will BTC be above $abc,def this week?")
        # Should NOT crash; returns unknown
        assert c.type == "unknown"

    def test_between_with_reversed_order(self):
        """If low > high, reject — malformed."""
        c = classify_title("Will BTC be between $90,000 and $80,000?")
        # We require low < high; this should fall through to unknown
        assert c.type != "between"
