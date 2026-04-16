"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class PolymarketConfig:
    wallet_private_key: str = ""
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"
    chain_id: int = 137  # Polygon


@dataclass(frozen=True)
class KalshiConfig:
    api_key: str = ""
    private_key_path: str = "kalshi_private_key.pem"
    base_url: str = "https://trading-api.kalshi.com"
    demo_url: str = "https://demo-api.kalshi.co"
    use_demo: bool = True


@dataclass(frozen=True)
class RiskConfig:
    kelly_fraction: float = 0.25
    max_position_size_pct: float = 0.03
    max_positions: int = 10
    min_edge: float = 0.05
    min_confidence: float = 0.45
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.15
    max_sector_concentration: float = 0.30
    stop_loss_pct: float = 0.10
    take_profit_pct: float = 0.25
    # Post-fee net edge threshold. Gross edge minus estimated round-trip fees
    # must exceed this. Default is stricter than min_edge because this is
    # the more important filter.
    min_net_edge: float = 0.03
    # Max positions within a correlated cluster (e.g. same-night NBA games).
    max_positions_per_cluster: int = 2
    # Use maker (post-only) limit orders at our predicted fair price.
    prefer_maker_orders: bool = True


@dataclass(frozen=True)
class CostConfig:
    daily_ai_cost_limit: float = 10.0
    per_decision_cap: float = 0.50
    analysis_cooldown_hours: float = 4.0
    max_analyses_per_market_per_day: int = 4


@dataclass(frozen=True)
class ScannerConfig:
    # Raised from $20k to $100k — $20k books have 1-3% slippage on entry
    min_volume: int = 100_000
    min_time_to_expiry_hours: int = 1
    max_time_to_expiry_days: int = 90
    min_price: float = 0.05
    max_price: float = 0.95
    max_candidates: int = 20
    interval_minutes: int = 5
    # Skip markets whose title/description matches ambiguous-resolution patterns
    skip_ambiguous_resolution: bool = True
    # Require at least this much USD depth near top of book (0 = disabled)
    min_book_depth: float = 0.0


@dataclass(frozen=True)
class EnsembleConfig:
    min_models_for_consensus: int = 3
    disagreement_threshold: float = 0.25
    disagreement_penalty: float = 0.30
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass(frozen=True)
class Settings:
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    openrouter_api_key: str = ""
    trading_mode: str = "paper"
    db_path: str = "trading.db"
    log_level: str = "INFO"


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


def load_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""
    load_dotenv()

    return Settings(
        polymarket=PolymarketConfig(
            wallet_private_key=_env("POLYGON_WALLET_PRIVATE_KEY"),
        ),
        kalshi=KalshiConfig(
            api_key=_env("KALSHI_API_KEY"),
            private_key_path=_env("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem"),
            use_demo=_env("KALSHI_USE_DEMO", "true").lower() == "true",
        ),
        risk=RiskConfig(
            kelly_fraction=_env_float("KELLY_FRACTION", 0.25),
            max_position_size_pct=_env_float("MAX_POSITION_SIZE_PCT", 0.03),
            max_positions=_env_int("MAX_POSITIONS", 10),
            min_edge=_env_float("MIN_EDGE", 0.05),
            min_net_edge=_env_float("MIN_NET_EDGE", 0.03),
            min_confidence=_env_float("MIN_CONFIDENCE", 0.45),
            max_daily_loss_pct=_env_float("MAX_DAILY_LOSS_PCT", 0.10),
            max_drawdown_pct=_env_float("MAX_DRAWDOWN_PCT", 0.15),
            max_sector_concentration=_env_float("MAX_SECTOR_CONCENTRATION", 0.30),
            max_positions_per_cluster=_env_int("MAX_POSITIONS_PER_CLUSTER", 2),
            prefer_maker_orders=_env("PREFER_MAKER_ORDERS", "true").lower() == "true",
        ),
        cost=CostConfig(
            # Accept either name — DAILY_AI_COST_LIMIT (canonical) or COST_MAX_DAILY_AI_USD (alias)
            daily_ai_cost_limit=_env_float(
                "DAILY_AI_COST_LIMIT",
                _env_float("COST_MAX_DAILY_AI_USD", 10.0),
            ),
        ),
        scanner=ScannerConfig(
            max_candidates=_env_int("SCANNER_MAX_CANDIDATES", 20),
            min_volume=_env_int("SCANNER_MIN_VOLUME", 20_000),
        ),
        ensemble=EnsembleConfig(
            temperature=_env_float("ENSEMBLE_TEMPERATURE", 0.3),
        ),
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        trading_mode=_env("TRADING_MODE", "paper"),
        db_path=_env("DB_PATH", "trading.db"),
        log_level=_env("LOG_LEVEL", "INFO"),
    )
