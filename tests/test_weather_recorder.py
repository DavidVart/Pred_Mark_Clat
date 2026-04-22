"""Tests for the weather-recorder's math and parsing logic."""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.weather_recorder.db import WeatherRecorderDB
from src.weather_recorder.recorder import (
    compute_our_prob,
    parse_book,
    parse_event_date,
    parse_strike,
    prob_leq,
)


class TestParseStrike:
    def test_parses_above_threshold(self):
        r = parse_strike("KXHIGHCHI-26APR23-T85", "86° or above")
        assert r == ("above", 85.0)

    def test_parses_below_threshold(self):
        r = parse_strike("KXHIGHNY-26APR22-T57", "56° or below")
        assert r == ("below", 57.0)

    def test_parses_between(self):
        r = parse_strike("KXHIGHNY-26APR23-B72.5", "72° to 73°")
        assert r == ("between", 72.5)

    def test_no_match_returns_none(self):
        assert parse_strike("SOMETHING-ELSE", "irrelevant") is None


class TestParseEventDate:
    @pytest.mark.parametrize("ticker,expected", [
        ("KXHIGHNY-26APR23", "2026-04-23"),
        ("KXHIGHLAX-25DEC01", "2025-12-01"),
        ("KXHIGHCHI-27JAN15-B80.5", "2027-01-15"),
    ])
    def test_parses_date(self, ticker, expected):
        assert parse_event_date(ticker) == expected

    def test_malformed_returns_none(self):
        assert parse_event_date("not-a-real-ticker") is None


class TestProbMath:
    def test_prob_leq_at_mean_is_half(self):
        assert abs(prob_leq(70, 70, sigma=3) - 0.5) < 1e-9

    def test_prob_leq_below_mean(self):
        p = prob_leq(65, 70, sigma=3)
        assert p < 0.5

    def test_prob_leq_above_mean(self):
        p = prob_leq(75, 70, sigma=3)
        assert p > 0.5

    def test_compute_below_direction(self):
        # Forecast 70°, threshold 65° (below), prob should be low
        p = compute_our_prob("below", 65, 70, 3)
        assert p < 0.2

    def test_compute_above_direction(self):
        # Forecast 70°, threshold 65° (above), prob should be high
        p = compute_our_prob("above", 65, 70, 3)
        assert p > 0.8

    def test_compute_between_with_bin_width(self):
        # Forecast 72°, bin centered at 72.5 with width 1 → peak density
        p = compute_our_prob("between", 72.5, 72, 3)
        # 1-bin at peak of N(0,3) density ≈ 0.13
        assert 0.10 < p < 0.17


class TestParseBook:
    def test_both_sides_present(self):
        book = {
            "yes_dollars": [["0.40", "100"], ["0.35", "200"]],
            "no_dollars":  [["0.55", "150"], ["0.50", "250"]],
        }
        bid, ask, depth = parse_book(book)
        assert bid == 0.40
        # ask = 1 - highest no bid = 1 - 0.55 = 0.45
        assert abs(ask - 0.45) < 1e-9
        assert depth == 700

    def test_empty_book(self):
        bid, ask, depth = parse_book({})
        assert bid is None and ask is None and depth == 0

    def test_yes_only(self):
        book = {"yes_dollars": [["0.30", "100"]]}
        bid, ask, depth = parse_book(book)
        assert bid == 0.30
        assert ask is None
        assert depth == 100


# --- DB ---

@pytest_asyncio.fixture
async def wr_db():
    db = WeatherRecorderDB(":memory:")
    await db.initialize()
    yield db
    await db.close()


class TestRecorderDB:
    @pytest.mark.asyncio
    async def test_log_and_read_snapshot(self, wr_db):
        await wr_db.log_snapshot({
            "market_ticker": "KXHIGHNY-26APR22-T57",
            "event_ticker": "KXHIGHNY-26APR22",
            "event_date": "2026-04-22",
            "city": "NYC",
            "direction": "below",
            "threshold": 57.0,
            "yes_best_bid": 0.58,
            "yes_best_ask": 0.60,
            "yes_mid": 0.59,
            "yes_book_depth": 500.0,
            "our_yes_prob": 0.70,
            "noaa_high_forecast": 55.0,
            "noaa_sigma": 3.0,
        })
        cursor = await wr_db.db.execute("SELECT * FROM wr_snapshots")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["city"] == "NYC"
        assert rows[0]["our_yes_prob"] == 0.70

    @pytest.mark.asyncio
    async def test_mark_resolved_updates(self, wr_db):
        await wr_db.log_snapshot({
            "market_ticker": "TEST-1", "event_ticker": "TEST",
            "event_date": "2026-04-22", "city": "NYC",
            "direction": "below", "threshold": 57.0,
            "our_yes_prob": 0.7,
        })
        count = await wr_db.mark_resolved("TEST-1", actual_high=55.0, outcome_yes=1)
        assert count == 1
        cursor = await wr_db.db.execute(
            "SELECT resolved, outcome_yes FROM wr_snapshots WHERE market_ticker='TEST-1'"
        )
        row = await cursor.fetchone()
        assert row["resolved"] == 1
        assert row["outcome_yes"] == 1
