"""Divergence signal detector.

Given:
    - Current BTC spot (from Coinbase websocket)
    - A Polymarket market quote (with opening reference price and time-to-resolution)
    - Our fair-value model

Emits a DivergenceSignal when:
    |fair_yes - market_yes| > threshold
    AND market is not too close to resolution (stale-quote zone)

The signal is DIRECTIONAL:
    gap > 0  → fair value higher than market → BUY YES
    gap < 0  → fair value lower than market → BUY NO

A SignalGate is also provided to deduplicate emissions — without it the
detector re-fires the same signal every cycle, which floods the DB and
would cause repeated trades in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.newsalpha.fair_value import FairValueParams, fair_yes_probability
from src.newsalpha.models import DivergenceSignal, MarketQuote
from src.utils.logging import get_logger

logger = get_logger("newsalpha.signal")


@dataclass(frozen=True)
class SignalConfig:
    # Minimum edge (fair vs market, absolute) to emit a signal
    min_edge: float = 0.03
    # Don't trade windows with less than this many seconds remaining
    min_seconds_remaining: float = 30.0
    # Don't trade windows with more than this many seconds remaining. Widened
    # to 4 hours so we can see the current 4h Polymarket "Up or Down" markets.
    # 5M markets naturally fall below this.
    max_seconds_remaining: float = 4 * 3600.0
    # If market doesn't publish a reference (opening) price, use current spot
    # at first observation (less accurate; flag in logs)
    use_current_as_reference_fallback: bool = True


def detect_divergence(
    quote: MarketQuote,
    spot: float,
    config: SignalConfig | None = None,
    fv_params: FairValueParams | None = None,
) -> DivergenceSignal | None:
    """Return a DivergenceSignal if the market mispricing is large enough.

    Does NOT trade — just emits the signal. Caller decides whether to act.
    """
    cfg = config or SignalConfig()

    sec_left = quote.seconds_remaining
    if sec_left < cfg.min_seconds_remaining or sec_left > cfg.max_seconds_remaining:
        return None

    # Establish the reference (strike) price
    ref = quote.starting_ref_price
    if ref is None:
        if not cfg.use_current_as_reference_fallback:
            return None
        # Fallback: use current spot as ref. Effectively says "will spot end
        # up higher than now" — still a tradeable question, just different.
        ref = spot
        logger.debug("ref_price_fallback", market=quote.market_id)

    window_total = (quote.window_end - quote.window_start).total_seconds() or 300.0
    fair_yes = fair_yes_probability(
        spot=spot,
        strike=ref,
        seconds_remaining=sec_left,
        window_total_seconds=window_total,
        params=fv_params,
    )

    gap = fair_yes - quote.yes_price

    if abs(gap) < cfg.min_edge:
        return None

    if gap > 0:
        # Market underpricing YES: buy YES at current price, fair says higher
        side = "yes"
        market_price = quote.yes_price
        fair_value = fair_yes
    else:
        # Market overpricing YES → buy NO
        side = "no"
        market_price = quote.no_price
        fair_value = 1.0 - fair_yes

    return DivergenceSignal(
        market_id=quote.market_id,
        title=quote.title,
        side=side,
        market_price=market_price,
        fair_value=fair_value,
        edge=abs(gap),
        seconds_remaining=sec_left,
        spot_reference=spot,
        spot_at_window_start=ref,
    )


class SignalGate:
    """Deduplicates signal emissions per market.

    Rules:
      - First signal on a given market always passes.
      - Subsequent signals on the same market are suppressed until EITHER:
          (a) cooldown_seconds have elapsed, OR
          (b) the signal's side flipped (yes → no or vice versa), indicating
              the market moved through fair value — always worth a fresh emit.
      - If a signal's edge grew significantly (>= edge_jump_threshold above
          last emitted), also pass — the thesis strengthened.

    Intended usage:
        gate = SignalGate(cooldown_seconds=300.0)
        if gate.should_emit(signal):
            db.log_signal(signal)
    """

    def __init__(
        self,
        cooldown_seconds: float = 300.0,
        edge_jump_threshold: float = 0.05,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.edge_jump_threshold = edge_jump_threshold
        self._last: dict[str, tuple[datetime, str, float]] = {}

    def should_emit(self, signal: DivergenceSignal) -> bool:
        key = signal.market_id
        prev = self._last.get(key)
        if prev is None:
            self._last[key] = (signal.timestamp, signal.side, signal.edge)
            return True

        prev_ts, prev_side, prev_edge = prev

        # Side flipped — always emit
        if signal.side != prev_side:
            self._last[key] = (signal.timestamp, signal.side, signal.edge)
            return True

        # Edge grew meaningfully — re-emit
        if signal.edge - prev_edge >= self.edge_jump_threshold:
            self._last[key] = (signal.timestamp, signal.side, signal.edge)
            return True

        # Cooldown elapsed — re-emit
        age = (signal.timestamp - prev_ts).total_seconds()
        if age >= self.cooldown_seconds:
            self._last[key] = (signal.timestamp, signal.side, signal.edge)
            return True

        return False

    def reset(self, market_id: str | None = None) -> None:
        """Clear dedup state — for a single market or all."""
        if market_id is None:
            self._last.clear()
        else:
            self._last.pop(market_id, None)
