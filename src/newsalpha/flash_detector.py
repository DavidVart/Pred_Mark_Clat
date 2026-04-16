"""Flash-move detector — catches sudden BTC price swings.

In steady state, Polymarket's market makers keep prices close to fair value.
Alpha lives in the 5–60 second window after a sudden price move, before
market makers fully reprice.

This module maintains a rolling buffer of BTC ticks and emits a "flash"
signal when the return over a short lookback exceeds a threshold. The
orchestrator uses flash state to:
    - Bypass the SignalGate cooldown (re-evaluate immediately)
    - Lower the min_edge threshold (tighter spreads are tradeable in fast markets)
    - Log the flash event for post-hoc analysis

Typical BTC statistics:
    - Median 1-min return: ~0.02%
    - 95th percentile 1-min return: ~0.15%
    - 99th percentile 1-min return: ~0.40%
    - "Flash" threshold at 0.20-0.30% catches 2-5 events/day on active days

The detector is called on every tick (via on_tick callback from CoinbaseWS).
It's O(1) per tick (amortized) via a deque with periodic cleanup.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from src.newsalpha.models import PriceTick
from src.utils.logging import get_logger

logger = get_logger("newsalpha.flash")


@dataclass
class FlashEvent:
    """A detected flash move."""

    symbol: str
    return_pct: float        # signed percentage return over lookback
    price_start: float       # price at start of lookback window
    price_end: float         # current price
    lookback_seconds: float  # actual seconds between start and end
    detected_at: float       # monotonic timestamp

    @property
    def direction(self) -> str:
        return "up" if self.return_pct > 0 else "down"


@dataclass
class FlashDetectorConfig:
    """Thresholds for flash detection."""

    # Lookback windows (seconds) to check returns over.
    # We check multiple windows — a 0.3% move in 30s is more impactful
    # than 0.3% over 120s, but both are worth noting.
    windows: list[float] = field(default_factory=lambda: [30.0, 60.0, 120.0])

    # Minimum absolute return (fraction, not percent) to qualify as a flash.
    # Default 0.002 = 0.2%. Roughly the 97th percentile of 1-min BTC returns.
    threshold: float = 0.002

    # After a flash is detected, how long (seconds) we remain in "flash active"
    # mode. During this window the orchestrator bypasses the signal gate and
    # lowers edge thresholds.
    flash_active_duration: float = 120.0

    # Maximum number of ticks to keep in the rolling buffer.
    # At ~1 tick/sec from Coinbase, 600 ticks = 10 minutes of history.
    max_buffer_size: int = 600


class FlashMoveDetector:
    """Detects sudden BTC price swings from a stream of PriceTicks.

    Usage:
        detector = FlashMoveDetector()
        coinbase_ws = CoinbaseTickerStream(on_tick=detector.on_tick)
        ...
        if detector.is_flash_active():
            # bypass gate, lower thresholds, etc.
    """

    def __init__(self, config: FlashDetectorConfig | None = None):
        self.config = config or FlashDetectorConfig()
        # Rolling buffer of (monotonic_time, price) tuples, per symbol.
        self._buffers: dict[str, deque[tuple[float, float]]] = {}
        # Latest flash event per symbol
        self._last_flash: dict[str, FlashEvent] = {}

    def on_tick(self, tick: PriceTick) -> None:
        """Called on every Coinbase tick. O(1) amortized."""
        now = time.monotonic()
        buf = self._buffers.setdefault(
            tick.symbol,
            deque(maxlen=self.config.max_buffer_size),
        )
        buf.append((now, tick.price))

        # Prune old entries beyond the longest window (deque maxlen handles
        # this automatically, but we also trim by time for correctness)
        max_window = max(self.config.windows)
        while buf and (now - buf[0][0]) > max_window * 1.5:
            buf.popleft()

        # Check returns over each window
        for window_sec in self.config.windows:
            ret = self._return_over(tick.symbol, window_sec, now)
            if ret is not None and abs(ret) >= self.config.threshold:
                # Find the starting price for logging
                start_price = self._price_at(tick.symbol, window_sec, now)
                event = FlashEvent(
                    symbol=tick.symbol,
                    return_pct=ret * 100.0,  # convert to percent for logging
                    price_start=start_price or tick.price,
                    price_end=tick.price,
                    lookback_seconds=window_sec,
                    detected_at=now,
                )
                prev = self._last_flash.get(tick.symbol)
                # Don't spam: only log a new flash if the previous one's active
                # period has expired OR direction changed.
                if prev is None or (now - prev.detected_at) > self.config.flash_active_duration or \
                        (prev.direction != event.direction):
                    self._last_flash[tick.symbol] = event
                    logger.info(
                        "flash_detected",
                        symbol=event.symbol,
                        direction=event.direction,
                        return_pct=f"{event.return_pct:+.3f}%",
                        window=f"{window_sec}s",
                        price_from=round(event.price_start, 2),
                        price_to=round(event.price_end, 2),
                    )

    def is_flash_active(self, symbol: str = "BTC-USD") -> bool:
        """True if a flash move was detected within the active window."""
        event = self._last_flash.get(symbol)
        if event is None:
            return False
        return (time.monotonic() - event.detected_at) < self.config.flash_active_duration

    def last_flash(self, symbol: str = "BTC-USD") -> FlashEvent | None:
        return self._last_flash.get(symbol)

    def current_return(self, symbol: str = "BTC-USD", seconds: float = 60.0) -> float | None:
        """Return the log-return over the last N seconds. None if insufficient data."""
        return self._return_over(symbol, seconds, time.monotonic())

    def _return_over(self, symbol: str, seconds: float, now: float) -> float | None:
        """Simple return (not log) over the last `seconds`."""
        buf = self._buffers.get(symbol)
        if not buf or len(buf) < 2:
            return None
        target_time = now - seconds
        # Binary search would be more efficient but deque is small (~600 entries)
        start_price = None
        for t, p in buf:
            if t >= target_time:
                start_price = p
                break
        if start_price is None or start_price == 0:
            return None
        end_price = buf[-1][1]
        return (end_price - start_price) / start_price

    def _price_at(self, symbol: str, seconds_ago: float, now: float) -> float | None:
        buf = self._buffers.get(symbol)
        if not buf:
            return None
        target = now - seconds_ago
        for t, p in buf:
            if t >= target:
                return p
        return None
