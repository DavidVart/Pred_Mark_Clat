"""Tests for market quality classification and cluster detection."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.models.market import UnifiedMarket
from src.pipeline.market_filters import (
    classify_market,
    cluster_key_from_title,
)


def make_market(title: str, description: str = "", category: str = "", platform: str = "polymarket") -> UnifiedMarket:
    return UnifiedMarket(
        platform=platform,
        market_id=f"m-{abs(hash(title)) % 10_000}",
        title=title,
        description=description,
        category=category,
        yes_price=0.5,
        no_price=0.5,
        volume=50_000,
        expiration=datetime.utcnow() + timedelta(days=10),
    )


class TestAmbiguousResolution:
    def test_ceasefire_market_flagged(self):
        m = make_market("Will there be a ceasefire in Ukraine by June 2026?")
        q = classify_market(m)
        assert q.is_ambiguous, f"Expected ambiguous, got: {q.reason}"

    def test_peace_deal_market_flagged(self):
        m = make_market("Will there be a peace deal signed by end of year?")
        q = classify_market(m)
        assert q.is_ambiguous

    def test_pardon_market_flagged(self):
        m = make_market("Will Snowden be pardoned by December 2026?")
        q = classify_market(m)
        assert q.is_ambiguous

    def test_meeting_market_flagged(self):
        m = make_market("Will Trump meet with Xi by March 2027?")
        q = classify_market(m)
        assert q.is_ambiguous

    def test_sports_market_not_flagged(self):
        m = make_market("Will the Lakers win the 2026 NBA Finals?")
        q = classify_market(m)
        assert not q.is_ambiguous

    def test_crypto_close_not_flagged(self):
        m = make_market("Will BTC close above $100,000 on Dec 31, 2026?")
        q = classify_market(m)
        assert not q.is_ambiguous


class TestCleanResolution:
    def test_sports_winner(self):
        m = make_market("Will the Lakers win the NBA Finals?")
        q = classify_market(m)
        assert q.is_clean

    def test_fed_rate_decision(self):
        m = make_market("Will the Federal Reserve cut rates in June 2026?")
        q = classify_market(m)
        assert q.is_clean

    def test_crypto_threshold(self):
        m = make_market("Will BTC close above $100k by year end?")
        q = classify_market(m)
        assert q.is_clean


class TestClustering:
    def test_nba_finals_all_teams_same_cluster(self):
        lakers = classify_market(make_market("Will Lakers win the 2026 NBA Finals?"))
        warriors = classify_market(make_market("Will Warriors win the 2026 NBA Finals?"))
        assert lakers.cluster == warriors.cluster == "nba:finals"

    def test_nhl_separate_from_nba(self):
        nba = classify_market(make_market("Will Lakers win the 2026 NBA Finals?"))
        nhl = classify_market(make_market("Will Oilers win the 2026 Stanley Cup?"))
        assert nba.cluster != nhl.cluster

    def test_fed_decisions_share_cluster(self):
        rate_cut = classify_market(make_market("Will the Federal Reserve cut rates?"))
        rate_hike = classify_market(make_market("Will the Fed hike rates?"))
        assert rate_cut.cluster == rate_hike.cluster

    def test_btc_markets_cluster_together(self):
        btc1 = classify_market(make_market("Will Bitcoin close above $100k?"))
        btc2 = classify_market(make_market("BTC above $120k by year end?"))
        assert btc1.cluster == btc2.cluster == "crypto:btc"


class TestClusterKeyFromTitle:
    def test_same_key_from_helper_and_classify(self):
        """The helper used by risk manager must match the scanner's classify."""
        m = make_market("Will the Boston Celtics win the 2026 NBA Finals?")
        from_classify = classify_market(m).cluster
        from_helper = cluster_key_from_title(m.title, m.category, m.platform)
        assert from_classify == from_helper
