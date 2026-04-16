"""Tests for the flash-move detector."""

from __future__ import annotations

import time

from src.newsalpha.flash_detector import FlashDetectorConfig, FlashMoveDetector
from src.newsalpha.models import PriceTick


def _tick(price: float, symbol: str = "BTC-USD") -> PriceTick:
    return PriceTick(symbol=symbol, price=price, source="coinbase")


class TestFlashMoveDetector:
    def test_no_flash_on_empty(self):
        d = FlashMoveDetector()
        assert not d.is_flash_active()

    def test_no_flash_on_flat_price(self):
        """Multiple ticks at the same price → no flash."""
        d = FlashMoveDetector(FlashDetectorConfig(threshold=0.002, windows=[1.0]))
        for _ in range(10):
            d.on_tick(_tick(100_000))
        assert not d.is_flash_active()

    def test_flash_on_sudden_drop(self):
        """Simulate a sharp 0.5% drop in under 1 second."""
        cfg = FlashDetectorConfig(
            threshold=0.002,   # 0.2%
            windows=[2.0],     # look at 2-second window
            flash_active_duration=5.0,
        )
        d = FlashMoveDetector(cfg)
        # Flat ticks
        for _ in range(5):
            d.on_tick(_tick(100_000))
        # Sudden drop
        d.on_tick(_tick(99_500))  # -0.5%
        assert d.is_flash_active()
        event = d.last_flash()
        assert event is not None
        assert event.direction == "down"
        assert abs(event.return_pct) > 0.2

    def test_flash_on_sudden_pump(self):
        cfg = FlashDetectorConfig(threshold=0.002, windows=[2.0], flash_active_duration=5.0)
        d = FlashMoveDetector(cfg)
        for _ in range(5):
            d.on_tick(_tick(100_000))
        d.on_tick(_tick(100_300))  # +0.3%
        assert d.is_flash_active()
        assert d.last_flash().direction == "up"

    def test_flash_expires(self):
        """Flash active period should expire after duration."""
        cfg = FlashDetectorConfig(
            threshold=0.002,
            windows=[2.0],
            flash_active_duration=0.1,  # 100ms for fast test
        )
        d = FlashMoveDetector(cfg)
        for _ in range(5):
            d.on_tick(_tick(100_000))
        d.on_tick(_tick(99_500))
        assert d.is_flash_active()
        time.sleep(0.15)
        assert not d.is_flash_active()

    def test_below_threshold_no_flash(self):
        """0.1% move with 0.2% threshold → no flash."""
        cfg = FlashDetectorConfig(threshold=0.002, windows=[2.0])
        d = FlashMoveDetector(cfg)
        for _ in range(5):
            d.on_tick(_tick(100_000))
        d.on_tick(_tick(99_900))  # -0.1%
        assert not d.is_flash_active()

    def test_current_return(self):
        cfg = FlashDetectorConfig(windows=[2.0])
        d = FlashMoveDetector(cfg)
        for _ in range(5):
            d.on_tick(_tick(100_000))
        d.on_tick(_tick(100_500))  # +0.5%
        ret = d.current_return("BTC-USD", seconds=2.0)
        assert ret is not None
        assert ret > 0

    def test_different_symbols_independent(self):
        cfg = FlashDetectorConfig(threshold=0.002, windows=[2.0])
        d = FlashMoveDetector(cfg)
        for _ in range(5):
            d.on_tick(_tick(100_000, "BTC-USD"))
            d.on_tick(_tick(3_000, "ETH-USD"))
        d.on_tick(_tick(99_500, "BTC-USD"))  # BTC drops
        assert d.is_flash_active("BTC-USD")
        assert not d.is_flash_active("ETH-USD")  # ETH unaffected
