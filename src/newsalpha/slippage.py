"""Realistic slippage simulator for Polymarket 5M/hourly BTC markets.

The paper-mode executor assumes we fill at the exact signal price. That's
wildly optimistic. Real fills have FOUR sources of degradation:

  1. LATENCY: between signal emission and order arrival at the exchange,
     BTC has moved. If our fair-value model depended on the old BTC
     price, the "edge" may be stale by the time the order lands.

  2. SPREAD CROSSING: taker orders pay the ask (buy) or hit the bid (sell).
     On thin 5M BTC books, the spread is typically 1-3 cents. Crossing
     eats half the spread on entry, half on exit.

  3. FILL PROBABILITY: limit orders (maker) don't always fill. Post-only
     limits at the bid fill 40-70% depending on market activity. Taker
     orders fill immediately but pay the spread + taker fee.

  4. FEES: Polymarket international charges 0.75-1.80% taker by category;
     0% maker. Already modeled in src/pipeline/fees.py.

This module implements a stochastic simulator that takes a signal + quote
and returns an ExecutionResult: did we fill? at what price? after what
latency? what fees? Deterministic with a seed for reproducible tests.

Design philosophy: be CONSERVATIVE (overestimate slippage). If post-friction
PnL is still positive, we have real edge. If it's negative, the paper
results were an illusion.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from src.newsalpha.models import DivergenceSignal, MarketQuote


Side = Literal["yes", "no"]
OrderType = Literal["taker", "maker"]


@dataclass(frozen=True)
class SlippageConfig:
    """Friction parameters — intentionally conservative (pessimistic)."""

    # --- Latency ---
    # Seconds between signal emission and order arrival at CLOB.
    # Polymarket CLOB is ~200-500ms API latency + our own pipeline overhead.
    latency_min_sec: float = 0.8
    latency_max_sec: float = 3.0

    # --- Price drift during latency ---
    # BTC 5-min realized vol — used to simulate price movement during latency.
    # σ_5min ≈ 0.3% is conservative (calm market); use 0.5% to stress test.
    btc_sigma_5min: float = 0.003

    # --- Spread crossing (price-scaled) ---
    # Real Polymarket spreads SCALE with price:
    #   - At mid ≈ 0.50, spreads are typically 1-3 cents (~2-6% of mid)
    #   - At mid ≈ 0.10, spreads are typically 0.5-1 cent (~5-10% of mid)
    #   - At mid ≈ 0.02, typical spread is 0.1-0.3 cents (tick-limited)
    # An ABSOLUTE 2-cent spread model would make OTM trades impossible
    # (100%+ slippage). We model spread as:
    #     spread = clamp(mid * pct, min_tick, max_absolute)
    # where pct has a random component for realism.
    spread_pct_of_mid_mean: float = 0.05    # 5% of mid
    spread_pct_of_mid_std: float = 0.02     # ±2% std dev
    min_tick: float = 0.001                 # Polymarket tick size (0.1 cent)
    max_absolute_spread: float = 0.05       # never more than 5c absolute

    # --- Fill probability ---
    # Taker order at ask → always fills (we're paying the spread)
    # Maker order at bid → fills probabilistically
    maker_fill_prob_at_bid: float = 0.55
    maker_fill_prob_at_bid_plus_1c: float = 0.75
    maker_timeout_sec: float = 30.0  # after this long, assume didn't fill

    # --- Fees ---
    # Polymarket int'l crypto category. Same taker on exit unless held to
    # resolution (NewsAlpha trades close within seconds/minutes — not held).
    taker_fee_pct: float = 0.018   # 1.80%
    maker_fee_pct: float = 0.0

    # --- Random seed for reproducibility ---
    random_seed: int | None = None


@dataclass
class ExecutionResult:
    """Output of simulating a fill attempt."""

    filled: bool = False
    reason: str = ""  # "filled" | "adverse_move" | "maker_not_filled" | "spread_too_wide"

    # If filled:
    fill_price: float | None = None
    fill_size: float | None = None
    fees_paid: float = 0.0
    effective_cost: float = 0.0  # fill_price * size + fees

    # Always populated:
    signal_price: float = 0.0       # the "paper" price we would have used
    quote_at_arrival_price: float = 0.0  # market ask at T+latency
    latency_ms: int = 0
    order_type: OrderType = "taker"

    # For accounting:
    slippage_bps: float = 0.0       # (fill - signal) / signal in basis points
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SlippageSimulator:
    """Stochastic simulator of real-world fill behavior for NewsAlpha signals.

    Usage:
        sim = SlippageSimulator(config=SlippageConfig())
        result = sim.simulate_entry(signal, quote, order_type="taker")
        if result.filled:
            # Use result.fill_price and result.effective_cost
    """

    def __init__(self, config: SlippageConfig | None = None):
        self.config = config or SlippageConfig()
        self._rng = random.Random(self.config.random_seed)

    def simulate_entry(
        self,
        signal: DivergenceSignal,
        quote: MarketQuote,
        order_type: OrderType = "taker",
    ) -> ExecutionResult:
        """Simulate a single entry attempt.

        Uses the signal's snapshot of market_price as T. Simulates:
          1. Latency L ~ uniform(latency_min, latency_max)
          2. Price drift during L based on BTC vol
          3. New quote at T+L
          4. Fill attempt with appropriate probability
        """
        cfg = self.config

        # 1. Latency
        latency_sec = self._rng.uniform(cfg.latency_min_sec, cfg.latency_max_sec)

        # 2. Price drift during latency — BTC moves randomly
        # σ(L) = σ_5min * sqrt(L/300s). That moves BTC by pct_drift.
        sigma_pct = cfg.btc_sigma_5min * math.sqrt(latency_sec / 300.0)
        btc_drift_pct = self._rng.gauss(0, sigma_pct)

        # Impact on market price: BTC up → YES price up (for "BTC higher" markets)
        # Approximate market_price move as proportional to fair-value sensitivity.
        # For a 5M BTC market with ~60% probability, d(yes_price)/d(btc_price) ≈ 0.3
        # per 1% BTC move near the strike. We use 0.3 as a conservative estimate.
        price_drift = btc_drift_pct * 0.3
        new_mid = signal.market_price + price_drift

        # Clamp to [0.01, 0.99]
        new_mid = max(0.01, min(0.99, new_mid))

        # 3. Spread at arrival time — SCALES with price (real Polymarket behavior)
        spread = self._sample_spread(new_mid)
        half_spread = spread / 2.0

        if signal.side == "yes":
            ask = new_mid + half_spread
            bid = new_mid - half_spread
        else:  # no side — spread symmetric around NO mid
            ask = new_mid + half_spread
            bid = new_mid - half_spread

        # Clamp bid/ask to valid probability range
        ask = max(cfg.min_tick, min(1.0 - cfg.min_tick, ask))
        bid = max(cfg.min_tick, min(1.0 - cfg.min_tick, bid))

        # 4. Fill simulation
        result = ExecutionResult(
            signal_price=signal.market_price,
            quote_at_arrival_price=new_mid,
            latency_ms=int(latency_sec * 1000),
            order_type=order_type,
        )

        if order_type == "taker":
            # Always fills — we cross the spread at ask
            fill_price = ask
            # Check if the move was adverse enough that we should refuse
            # (e.g., signal said buy YES at 0.40 with fair 0.55; now ask is 0.60
            # → "edge" evaporated, don't fill)
            signal_fair = signal.fair_value
            edge_at_fill = signal_fair - fill_price
            if edge_at_fill <= 0:
                result.filled = False
                result.reason = "adverse_move"
                result.slippage_bps = (fill_price - signal.market_price) / signal.market_price * 10000
                return result

            # Taker fee
            fees_pct = cfg.taker_fee_pct
            result.filled = True
            result.reason = "filled"
            result.fill_price = fill_price
            result.fees_paid = 0.0  # will be computed by caller with size
            result.slippage_bps = (fill_price - signal.market_price) / signal.market_price * 10000
            # effective_cost and fill_size set by caller who knows the size

        else:  # maker
            # Post limit at bid (or bid+1c for higher fill prob)
            limit_price = bid  # post at the bid

            # Fill probability
            fill_prob = cfg.maker_fill_prob_at_bid
            if self._rng.random() > fill_prob:
                result.filled = False
                result.reason = "maker_not_filled"
                result.slippage_bps = 0.0
                return result

            # Filled at the bid — saved a full spread
            fill_price = limit_price
            signal_fair = signal.fair_value
            edge_at_fill = signal_fair - fill_price
            if edge_at_fill <= 0:
                result.filled = False
                result.reason = "adverse_move"
                return result

            result.filled = True
            result.reason = "filled"
            result.fill_price = fill_price
            result.fees_paid = 0.0  # maker = 0% fee
            result.slippage_bps = (fill_price - signal.market_price) / signal.market_price * 10000

        return result

    def apply_fee(self, result: ExecutionResult, size: float) -> None:
        """Fill in fee + effective_cost now that size is known."""
        if not result.filled or result.fill_price is None:
            return
        cfg = self.config
        notional = result.fill_price * size
        fees_pct = cfg.taker_fee_pct if result.order_type == "taker" else cfg.maker_fee_pct
        result.fees_paid = notional * fees_pct
        result.fill_size = size
        result.effective_cost = notional + result.fees_paid

    def _sample_spread(self, mid: float) -> float:
        """Sample a realistic bid-ask spread at the given mid-price.

        Spread = max(min_tick, clamp(mid * pct, min_tick, max_absolute))
        where pct is drawn from Normal(mean, std) and floored at 0.
        """
        cfg = self.config
        pct = max(0.0, self._rng.gauss(cfg.spread_pct_of_mid_mean, cfg.spread_pct_of_mid_std))
        raw = mid * pct
        spread = max(cfg.min_tick, min(cfg.max_absolute_spread, raw))
        return spread

    def simulate_exit(
        self,
        exit_market_price: float,
        side: Side,
        order_type: OrderType = "taker",
    ) -> tuple[float, float]:
        """Simulate exit fill: returns (fill_price, fee_pct).

        Uses the same price-scaled spread model as entry. When CLOSING we
        receive the bid (half-spread below mid).
        """
        cfg = self.config
        spread = self._sample_spread(exit_market_price)
        half_spread = spread / 2.0
        fill_price = max(cfg.min_tick, exit_market_price - half_spread)
        fee_pct = cfg.taker_fee_pct if order_type == "taker" else cfg.maker_fee_pct
        return fill_price, fee_pct
