"""Filters for market quality — resolution ambiguity, correlated clusters, categorization.

Based on observed failure modes in prediction markets:
- Ambiguous resolution criteria (e.g. Polymarket $7M Ukraine mineral deal dispute, March 2025)
- Subjective outcomes requiring human judgment (opinion polls, "agreed to" language)
- Vague date boundaries ("by end of year" vs specific deadline)

Whitelisted market types have clean, mechanical resolution sources:
- Sports game outcomes (official league results)
- Crypto price-at-close (on-chain oracle)
- Election state calls (AP/major networks)
- Fed rate decisions (FOMC announcement)
- Economic data releases (BLS, BEA)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.market import UnifiedMarket


# Phrases that correlate with ambiguous UMA/manual resolution disputes.
# If a market title or description contains any of these, we skip it.
AMBIGUOUS_RESOLUTION_PATTERNS = [
    r"\bdeal\b.*\bby\b",              # "deal by X date" — what counts as a deal?
    r"\bagree(s|ment|d)?\b.*\bby\b",   # "agreement by X" — oral? signed? MOU?
    r"\bannounce(s|d|ment)\b.*\bby\b", # "announcement" — press release? tweet?
    r"\bsign(s|ed|ing)\b.*\bdeal\b",
    r"\bnegotiat(e|ed|ions?)\b",
    r"\bverif(y|ied|ication)\b.*\bby\b",
    r"\bofficial(ly)? recogniz",
    r"\bpeace (deal|agreement|treaty)\b",
    r"\bceasefire\b",
    r"\bresign(s|ation)?\b.*\bby\b",
    r"\bpardon(s|ed)?\b.*\bby\b",       # "pardoned by X" — commutation? full?
    r"\bmeet(s|ing)?\b.*\bwith\b.*\bby\b",  # "meet with X by Y" — phone call? in person?
]

# Keywords that mark clean-resolution markets — we positively prefer these.
CLEAN_RESOLUTION_KEYWORDS = [
    "win the",
    "score",
    "points",
    "close above",
    "close below",
    "above $",
    "below $",
    "federal reserve",
    "fomc",
    "rate cut",
    "rate hike",
    "cpi",
    "unemployment rate",
    "nonfarm",
    "electoral",
    "popular vote",
    "percent of the vote",
]


@dataclass(frozen=True)
class MarketQuality:
    is_ambiguous: bool
    is_clean: bool
    cluster: str  # correlated grouping key, e.g. "nba:20260601"
    reason: str


def classify_market(market: UnifiedMarket) -> MarketQuality:
    """Classify a market's resolution quality and correlation cluster."""
    text = f"{market.title} {market.description}".lower()

    # Ambiguity check — any dispute-prone pattern
    for pattern in AMBIGUOUS_RESOLUTION_PATTERNS:
        if re.search(pattern, text):
            return MarketQuality(
                is_ambiguous=True,
                is_clean=False,
                cluster=_infer_cluster(market, text),
                reason=f"ambiguous_resolution: matched /{pattern}/",
            )

    # Clean positive signal
    is_clean = any(kw in text for kw in CLEAN_RESOLUTION_KEYWORDS)

    return MarketQuality(
        is_ambiguous=False,
        is_clean=is_clean,
        cluster=_infer_cluster(market, text),
        reason="clean" if is_clean else "unclassified",
    )


def _infer_cluster(market: UnifiedMarket, text_lower: str) -> str:
    """Derive a correlation cluster key.

    Examples:
    - "NBA Finals" + all 30 teams → cluster = "nba:finals"
    - "Stanley Cup" + all 32 teams → cluster = "nhl:stanley_cup"
    - NFL Week 5 games → cluster = "nfl:20260925"  (same date)
    - Fed rate June 2026 → cluster = "fed:20260619"
    - Presidential election 2028 → cluster = "election:2028:president"
    - Anything else → cluster = f"{platform}:other"

    Markets sharing a cluster key are treated as correlated — we cap exposure
    across the cluster, not per-market.
    """
    t = text_lower

    # Major sports leagues
    if "nba finals" in t or "nba championship" in t:
        return "nba:finals"
    if "nba rookie of the year" in t or "nba roy" in t:
        return "nba:awards"
    if "stanley cup" in t or "nhl champion" in t:
        return "nhl:stanley_cup"
    if "world series" in t or "mlb champion" in t:
        return "mlb:world_series"
    if "super bowl" in t or "nfl champion" in t:
        return "nfl:super_bowl"
    if "mvp" in t:
        return f"{_detect_league(t)}:mvp"

    # Politics
    if "federal reserve" in t or "fomc" in t or "fed " in t or "rate cut" in t or "rate hike" in t:
        return "fed:rate_decision"
    if "2028 election" in t or "2028 presidential" in t:
        return "election:2028"
    if "midterm" in t:
        return "election:midterm"

    # Crypto
    if "bitcoin" in t or "btc" in t:
        return "crypto:btc"
    if "ethereum" in t or "eth " in t or " eth?" in t:
        return "crypto:eth"

    # Fall back to category + platform
    if market.category:
        return f"{market.platform}:{market.category.lower()}"
    return f"{market.platform}:other"


def _detect_league(text_lower: str) -> str:
    for league in ("nba", "nhl", "nfl", "mlb", "mls", "ufc"):
        if league in text_lower:
            return league
    return "sports"


def cluster_key_from_title(title: str, category: str, platform: str) -> str:
    """Cluster key for existing positions where we only have title/category/platform.

    Mirrors _infer_cluster() but operates on raw strings. Used by the risk manager
    when counting existing positions in a correlated cluster.
    """
    # Build a dummy market object just for _infer_cluster's signature.
    # We avoid importing UnifiedMarket here to keep this module light.
    class _Stub:
        def __init__(self, t: str, c: str, p: str):
            self.title = t
            self.description = ""
            self.category = c
            self.platform = p

    stub = _Stub(title or "", category or "", platform or "")
    return _infer_cluster(stub, f"{stub.title} {stub.description}".lower())
